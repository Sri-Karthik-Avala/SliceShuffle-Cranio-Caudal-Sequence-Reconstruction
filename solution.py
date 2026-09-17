"""
SliceShuffle: Cranio-Caudal Sequence Reconstruction — solution.

Pipeline (self-contained, trains from scratch; no external weights, no network):
  * Shared flip-invariant CNN encoder -> permutation-equivariant set-transformer
    contextualizes the 32 shuffled slices of a volume.
  * pos head (scalar) + pairwise ranking loss  -> cranio-caudal ORDER (pred_rank).
  * cls head (48-way, soft Gaussian targets)   -> per-slice absolute-position potential.
  * gap head (ordinal over consecutive ordered slices) + lead/trail edge heads -> gap sizes.
  * A monotone "unified" DP assigns the 32 ordered slices to 32 of 48 integer positions,
    anchored by the cls potentials and shaped by the gap head + empirical edge/gap priors;
    the 16 unused positions are the predicted missing mask (always exactly 16, identical on
    every row of a volume).

Reads  : <root>/train.csv, <root>/test.csv, <root>/train/<vol>/sNN.png, <root>/test/<vol>/sNN.png
         where <root> is auto-detected among {'', 'public/', 'dataset/public/', 'input/', ...}.
Writes : working/submission.csv
"""
import os, sys, math, time, copy, glob, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GMAX = 10; INF = 1e15
SEED0 = 0

# ---------------------------------------------------------------- paths / data
def find_root():
    cands = ['', 'public/', 'dataset/public/', 'input/', '../input/', './data/', 'data/']
    for r in cands:
        if os.path.exists(r+'train.csv') and os.path.exists(r+'test.csv') and os.path.isdir(r+'train') and os.path.isdir(r+'test'):
            return r
    # last resort: search
    for base in ['.', '..', 'dataset', 'public']:
        for dp, dns, fns in os.walk(base):
            if 'train.csv' in fns and 'test.csv' in fns and os.path.isdir(os.path.join(dp, 'train')):
                return dp.rstrip('/\\')+'/' if not dp.endswith(('/', '\\')) else dp
    raise FileNotFoundError("could not locate train.csv/test.csv + train/ test/ dirs")

def folder_map(split_dir):
    m = {}
    for name in os.listdir(split_dir):
        full = os.path.join(split_dir, name)
        if os.path.isdir(full):
            try: m[int(name)] = full
            except ValueError: m[name] = full
    return m

def load_split(root, split, vols):
    sd = root+split; fm = folder_map(sd)
    arr = np.zeros((len(vols), 32, 192, 192), np.uint8)
    for vi, v in enumerate(vols):
        d = fm.get(int(v)) or fm.get(str(v))
        for s in range(32):
            arr[vi, s] = np.asarray(Image.open(os.path.join(d, f's{s:02d}.png')).convert('L').resize((192, 192)), np.uint8)
    return arr

def build_targets(df, vols):
    pos = np.zeros((len(vols), 32), np.int64); rank = np.zeros((len(vols), 32), np.int64); mask = np.zeros((len(vols), 48), np.int64)
    for vi, v in enumerate(vols):
        sub = df[df['volume_id'] == v].sort_values('presented_index')
        m = np.array([int(c) for c in str(sub['true_missing_mask'].iloc[0])[1:]]); mask[vi] = m
        present = np.where(m == 0)[0]; r = sub['true_rank'].to_numpy(); rank[vi] = r
        L = len(present); idx = np.round(r*(L-1)/31.0).astype(int).clip(0, L-1); pos[vi] = present[idx]
    return pos, rank, mask

# ---------------------------------------------------------------- model
class Res(nn.Module):
    def __init__(s, ci, co, stride):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(co)
        s.sc = nn.Sequential() if (stride == 1 and ci == co) else nn.Sequential(nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))
    def forward(s, x):
        y = F.relu(s.b1(s.c1(x)), True); y = s.b2(s.c2(y)); return F.relu(y+s.sc(x), True)

class Encoder(nn.Module):
    def __init__(s, d=384):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, 48, 3, 2, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True))
        s.l1 = nn.Sequential(Res(48, 64, 2), Res(64, 64, 1)); s.l2 = nn.Sequential(Res(64, 128, 2), Res(128, 128, 1))
        s.l3 = nn.Sequential(Res(128, 256, 2), Res(256, 256, 1)); s.l4 = nn.Sequential(Res(256, d, 2), Res(d, d, 1))
    def forward(s, x):
        x = s.stem(x); x = s.l1(x); x = s.l2(x); x = s.l3(x); x = s.l4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

class Net(nn.Module):
    def __init__(s, denc=384, d=256, layers=3, heads=4):
        super().__init__()
        s.enc = Encoder(denc); s.proj = nn.Linear(denc, d)
        enc = nn.TransformerEncoderLayer(d, heads, d*2, 0.1, 'gelu', batch_first=True, norm_first=True)
        s.tx = nn.TransformerEncoder(enc, layers); s.norm = nn.LayerNorm(d)
        s.pos_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        s.cls_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 48))
        s.gru = nn.GRU(d, d//2, batch_first=True, bidirectional=True)
        s.gap_head = nn.Sequential(nn.Linear(d*4, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, GMAX))
        s.lead_head = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, GMAX))
        s.trail_head = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, GMAX)); s.d = d
    def embed(s, x, flip_tta=False):
        B, T = x.shape[:2]; xf = x.reshape(B*T, *x.shape[2:]); e = s.enc(xf)
        if flip_tta: e = 0.5*(e+s.enc(torch.flip(xf, dims=[3])))
        return s.norm(s.tx(s.proj(e.reshape(B, T, -1))))
    def pos(s, h): return s.pos_head(h).squeeze(-1)
    def cls(s, h): return s.cls_head(h)
    def gaps(s, h, order):
        c = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, s.d)); g, _ = s.gru(c)
        a, b = g[:, :-1], g[:, 1:]; feat = torch.cat([a, b, a-b, (a-b).abs()], -1); ctx = g.mean(1)
        return s.gap_head(feat), s.lead_head(torch.cat([g[:, 0], ctx], -1)), s.trail_head(torch.cat([g[:, -1], ctx], -1))

def augment(x):
    B, T, _, R, _ = x.shape; n = B*T; x = x.reshape(n, 1, R, R)
    flip = torch.rand(n, device=x.device) < 0.5; x[flip] = torch.flip(x[flip], dims=[3])
    ang = (torch.rand(n, device=x.device)*2-1)*(10*math.pi/180); tx = (torch.rand(n, device=x.device)*2-1)*0.05; ty = (torch.rand(n, device=x.device)*2-1)*0.05
    sc = 1+(torch.rand(n, device=x.device)*2-1)*0.06; cos, sin = torch.cos(ang)*sc, torch.sin(ang)*sc
    th = torch.zeros(n, 2, 3, device=x.device); th[:, 0, 0]=cos; th[:, 0, 1]=-sin; th[:, 0, 2]=tx; th[:, 1, 0]=sin; th[:, 1, 1]=cos; th[:, 1, 2]=ty
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False, padding_mode='zeros')
    gm = (0.8+0.45*torch.rand(n, 1, 1, 1, device=x.device)); x = x.clamp(1e-4, 1).pow(gm)
    x = x*(0.85+0.3*torch.rand(n, 1, 1, 1, device=x.device)) + (torch.rand(n, 1, 1, 1, device=x.device)*2-1)*0.05
    return (x+torch.randn_like(x)*0.03).clamp(0, 1).reshape(B, T, 1, R, R)

def pairwise_rank_loss(sc, tgt):
    d = sc.unsqueeze(2)-sc.unsqueeze(1); t = tgt.unsqueeze(2)-tgt.unsqueeze(1); m = (t != 0).float()
    return (F.softplus(-torch.sign(t)*d)*m).sum()/m.sum().clamp(min=1)

def make_soft_targets(pos, sigma=1.2):
    j = np.arange(48)[None, None, :]; t = pos[:, :, None]; w = np.exp(-0.5*((j-t)/sigma)**2)
    return (w/w.sum(-1, keepdims=True)).astype(np.float32)

# ---------------------------------------------------------------- decode
def macro_f1(p, t): return f1_score(t, p, average='macro', labels=[0, 1], zero_division=0)
def logsm1(x): x = x-x.max(); e = np.exp(x); return np.log(e/e.sum()+1e-12)
def logsm2(x): x = x-x.max(1, keepdims=True); e = np.exp(x); return np.log(e/e.sum(1, keepdims=True)+1e-12)
_A = 48; _ii = np.arange(_A); _D = (_ii[None, :]-_ii[:, None]-1).clip(0, _A-1); _UP = _ii[None, :] > _ii[:, None]
def _tailvec(arr):
    out = np.empty(_A); out[:GMAX] = arr[:GMAX]
    for g in range(GMAX, _A): out[g] = arr[GMAX-1]-(g-GMAX+1)*2.0
    return out

def unified_dp(cls_lp, gli, leadlp, traillp, pri, w_cls, w_gap, w_iprior, w_lead, w_lprior, w_occ, lp_present):
    A = _A
    gc = -w_gap*np.stack([_tailvec(gli[v]) for v in range(31)]) - w_iprior*_tailvec(pri['internal'])[None, :]
    node = -w_cls*cls_lp - w_occ*lp_present[None, :]
    leadv = _tailvec(leadlp); trailv = _tailvec(traillp); lpri = _tailvec(pri['lead']); tpri = _tailvec(pri['trail'])
    dp = node[0] - w_lead*leadv - w_lprior*lpri; back = np.zeros((32, A), int)
    for v in range(1, 32):
        gmat = gc[v-1][_D].copy(); gmat[~_UP] = INF
        M = dp[:, None] + gmat; prev = np.argmin(M, axis=0); dp = M[prev, _ii] + node[v]
        if v == 31: dp = dp - w_lead*trailv[47-_ii] - w_lprior*tpri[47-_ii]
        back[v] = prev
    a = int(np.argmin(dp)); slots = []
    for v in range(31, -1, -1): slots.append(a); a = back[v, a]
    m = np.ones(48, int); m[slots] = 0
    return m  # always exactly 16 ones (32 strictly-increasing slots assigned)

def decode_all(POS, CLS, GL, LE, TR, pri, lp_present, W):
    out = []
    for i in range(len(POS)):
        o = np.argsort(POS[i]); cls_lp = logsm2(CLS[i])[o]
        out.append(unified_dp(cls_lp, logsm2(GL[i]), logsm1(LE[i]), logsm1(TR[i]), pri, **W, lp_present=lp_present))
    return out

# ---------------------------------------------------------------- train / predict
def get_batch(img, ii, R):
    x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
    if R != 192:
        b, t = x.shape[:2]; x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(R, R), mode='bilinear', align_corners=False).reshape(b, t, 1, R, R)
    return x

@torch.no_grad()
def predict(model, img, idx, R, flip_tta=True, bs=8):
    model.eval(); POS=[]; CLS=[]; GL=[]; LE=[]; TR=[]
    for s in range(0, len(idx), bs):
        ii = idx[s:s+bs]; x = get_batch(img, ii, R); h = model.embed(x, flip_tta=flip_tta)
        pos = model.pos(h).cpu().numpy(); cls = model.cls(h).cpu().numpy()
        order = np.argsort(pos, axis=1); gl, le, tr = model.gaps(h, torch.from_numpy(order).to(DEV))
        POS.append(pos); CLS.append(cls); GL.append(gl.cpu().numpy()); LE.append(le.cpu().numpy()); TR.append(tr.cpu().numpy())
    return [np.concatenate(POS), np.concatenate(CLS), np.concatenate(GL), np.concatenate(LE), np.concatenate(TR)]

def train_one(seed, img, pos, rank, trn_idx, cls_w, soft, R, epochs, lr=2e-3, bs=8, ema_decay=0.998):
    torch.manual_seed(seed); np.random.seed(seed)
    model = Net().to(DEV); ema = copy.deepcopy(model); [p.requires_grad_(False) for p in ema.parameters()]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs*(len(trn_idx)//bs+1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.08)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEV == 'cuda'))
    post = torch.from_numpy(pos).to(DEV).float(); rankt = torch.from_numpy(rank).to(DEV); softt = torch.from_numpy(soft).to(DEV)
    idx = trn_idx.copy(); gstep = 0
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            ii = idx[s:s+bs]; x = augment(get_batch(img, ii, R)); tgt = post[ii]; rk = rankt[ii]; order = torch.argsort(rk, dim=1)
            with torch.cuda.amp.autocast(enabled=(DEV == 'cuda')):
                h = model.embed(x); pscore = model.pos(h); clsl = model.cls(h)
                reg = F.smooth_l1_loss(pscore, tgt/47.0); rkl = pairwise_rank_loss(pscore, tgt)
                cls_loss = -(softt[ii]*F.log_softmax(clsl, -1)).sum(-1).mean()
                gl, lead, trail = model.gaps(h, order); pos_ord = torch.gather(tgt, 1, order)
                gtar = (pos_ord[:, 1:]-pos_ord[:, :-1]-1).clamp(0, GMAX-1).long()
                ltar = pos_ord[:, 0].clamp(0, GMAX-1).long(); ttar = (47-pos_ord[:, 31]).clamp(0, GMAX-1).long()
                gap_loss = F.cross_entropy(gl.reshape(-1, GMAX), gtar.reshape(-1), weight=cls_w, label_smoothing=0.05)
                edge_loss = F.cross_entropy(lead, ltar, label_smoothing=0.05)+F.cross_entropy(trail, ttar, label_smoothing=0.05)
                loss = 1.0*reg + 1.0*rkl + 1.0*cls_loss + 0.3*gap_loss + 0.1*edge_loss
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            gstep += 1; dec = min(ema_decay, (1+gstep)/(10+gstep))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(dec).add_(pm, alpha=1-dec)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
    return ema

def ens_predict(models, img, idx):
    # models: list of (model, R) — each predicted at its own resolution, logits averaged (multi-scale)
    accP = accC = accG = accL = accT = None
    for m, R in models:
        P, C, G, L, T = predict(m, img, idx, R)
        if accP is None: accP, accC, accG, accL, accT = P, C, G, L, T
        else: accP, accC, accG, accL, accT = accP+P, accC+C, accG+G, accL+L, accT+T
    n = len(models)
    return [accP/n, accC/n, accG/n, accL/n, accT/n]

def build_priors(tm):
    lead_c = np.zeros(GMAX); trail_c = np.zeros(GMAX); int_c = np.zeros(GMAX)
    for r in tm:
        p = np.where(r == 0)[0]; lead_c[min(p[0], GMAX-1)] += 1; trail_c[min(47-p[-1], GMAX-1)] += 1
        for g in np.diff(p)-1: int_c[min(max(g, 0), GMAX-1)] += 1
    pri = {'lead': np.log(lead_c/lead_c.sum()+1e-9), 'trail': np.log(trail_c/trail_c.sum()+1e-9), 'internal': np.log(int_c/int_c.sum()+1e-9)}
    int_w = np.clip(int_c.sum()/(GMAX*(int_c+1)), 0.2, 8.0)
    lp_present = np.log((1-tm.mean(0)).clip(.02, .98)/tm.mean(0).clip(.02, .98))
    return pri, int_w, lp_present

WGRID = [dict(w_cls=wc, w_gap=wg, w_iprior=wi, w_lead=wl, w_lprior=wp, w_occ=wo)
         for wc in [0.5, 1.0, 2.0, 3.0, 4.0] for wg in [0.5, 1.0, 2.0] for wi in [0.2, 0.4]
         for wl in [0.0, 1.0] for wp in [0.5, 1.5] for wo in [0.0, 0.5]]

def tune_W(preds, rank, mask, pri, lp_present):
    # split-robust W selection: maximize the worse of two holdout halves (generalizes better than argmax-full)
    POS = preds[0]; n = len(POS); taus = [max(0, kendalltau(np.argsort(np.argsort(POS[i])), rank[i]).statistic)**2 for i in range(n)]
    def mask2(idxs, W):
        ms = []
        for i in idxs:
            o = np.argsort(POS[i]); ms.append(unified_dp(logsm2(preds[1][i])[o], logsm2(preds[2][i]), logsm1(preds[3][i]), logsm1(preds[4][i]), pri, **W, lp_present=lp_present))
        return float(np.mean([macro_f1(ms[j], mask[idxs[j]])**2 for j in range(len(idxs))]))
    half = n//2; A = list(range(half)); B = list(range(half, n))
    W = max(WGRID, key=lambda w: min(mask2(A, w), mask2(B, w)))
    return float(np.mean(taus)), mask2(list(range(n)), W), W

# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    root = find_root(); print("root:", repr(root), "device:", DEV, flush=True)
    tr_df = pd.read_csv(root+'train.csv'); tr_df['volume_id'] = tr_df['volume_id'].astype(int)
    te_df = pd.read_csv(root+'test.csv'); te_df['volume_id'] = te_df['volume_id'].astype(int)
    tr_vols = sorted(tr_df['volume_id'].unique()); te_vols = sorted(te_df['volume_id'].unique())
    print(f"train vols {len(tr_vols)} test vols {len(te_vols)}", flush=True)

    img = load_split(root, 'train', tr_vols); pos, rank, mask = build_targets(tr_df, tr_vols)
    te_img = load_split(root, 'test', te_vols)
    print(f"loaded images {time.time()-t0:.0f}s", flush=True)

    # config (GPU vs CPU); env-overridable for sandbox time budgets.
    # Multi-scale ensemble (160 + 192) x N seeds = the validated winning config (honest holdout S~0.823).
    if DEV == 'cuda':
        EPOCHS = int(os.environ.get('SS_EPOCHS', 64)); nseed = int(os.environ.get('SS_SEEDS', 3))
        scales = [int(x) for x in os.environ.get('SS_SCALES', '160,192').split(',')]
        CFG = [(R, sd) for R in scales for sd in range(nseed)]
    else:
        EPOCHS = int(os.environ.get('SS_EPOCHS', 14)); CFG = [(112, sd) for sd in range(int(os.environ.get('SS_SEEDS', 1)))]

    N = len(tr_vols); rng = np.random.RandomState(0); perm = rng.permutation(N)
    n_hold = int(os.environ.get('SS_HOLD', max(80, N//6))); hold_idx = perm[:n_hold]; trn_idx = perm[n_hold:]

    # priors from the training portion only (used during W-tuning); priors are near-identical on full set
    pri, int_w, lp_present = build_priors(mask[trn_idx]); cls_w = torch.tensor(int_w, dtype=torch.float32, device=DEV)
    soft = make_soft_targets(pos)

    # train multi-scale ensemble on trn_idx (all-but-holdout); tune decode W on the holdout; reuse models for test
    models = []
    for (R, sd) in CFG:
        m = train_one(sd, img, pos, rank, trn_idx, cls_w, soft, R, EPOCHS); models.append((m, R))
        print(f"trained R{R} seed {sd}  {time.time()-t0:.0f}s", flush=True)
    hpred = ens_predict(models, img, hold_idx)
    r2, msq, W = tune_W(hpred, rank[hold_idx], mask[hold_idx], pri, lp_present)
    print(f"HOLDOUT rank2={r2:.4f} mask2={msq:.4f} S={0.6*r2+0.4*msq:.4f} W={W}", flush=True)

    # full-data priors for final test decoding
    pri_all, _, lp_present_all = build_priors(mask)
    tpred = ens_predict(models, te_img, np.arange(len(te_vols)))
    # save logits for offline decode verification (optional, harmless)
    try:
        np.savez_compressed('_logits.npz',
            hP=hpred[0], hC=hpred[1], hG=hpred[2], hL=hpred[3], hT=hpred[4],
            tP=tpred[0], tC=tpred[1], tG=tpred[2], tL=tpred[3], tT=tpred[4],
            hold_rank=rank[hold_idx], hold_mask=mask[hold_idx],
            pri_lead=pri_all['lead'], pri_trail=pri_all['trail'], pri_internal=pri_all['internal'],
            lp_present=lp_present_all, te_vols=np.array(te_vols))
    except Exception as _e:
        print("logit save skipped:", _e, flush=True)
    POS = tpred[0]; te_masks = decode_all(*tpred, pri_all, lp_present_all, W)
    pred_rank = {te_vols[i]: np.argsort(np.argsort(POS[i])) for i in range(len(te_vols))}
    mask_str = {te_vols[i]: 'M'+''.join(str(int(b)) for b in te_masks[i]) for i in range(len(te_vols))}
    for v in te_vols: assert te_masks[te_vols.index(v)].sum() == 16, v

    # write submission matching test.csv exactly
    rows = []
    for _, r in te_df.sort_values(['volume_id', 'presented_index']).iterrows():
        v = int(r['volume_id']); pidx = int(r['presented_index'])
        rows.append((int(r['row_id']), v, pidx, int(pred_rank[v][pidx]), mask_str[v]))
    out = pd.DataFrame(rows, columns=['row_id', 'volume_id', 'presented_index', 'pred_rank', 'pred_missing_mask'])
    out = out.sort_values('row_id').reset_index(drop=True)
    os.makedirs('working', exist_ok=True); out.to_csv('working/submission.csv', index=False)
    print(f"wrote working/submission.csv rows={len(out)}  total {time.time()-t0:.0f}s", flush=True)

def fallback():
    """Never-crash path: presented-order ranks + constant top-16 prior mask."""
    root = find_root()
    tr_df = pd.read_csv(root+'train.csv'); tr_df['volume_id'] = tr_df['volume_id'].astype(int)
    te_df = pd.read_csv(root+'test.csv'); te_df['volume_id'] = te_df['volume_id'].astype(int)
    tr_vols = sorted(tr_df['volume_id'].unique()); _, _, mask = build_targets(tr_df, tr_vols)
    marg = mask.mean(0); pm = np.zeros(48, int); pm[np.argsort(-marg)[:16]] = 1
    ms = 'M'+''.join(str(int(b)) for b in pm)
    out = te_df[['row_id', 'volume_id', 'presented_index']].copy()
    out['pred_rank'] = out['presented_index'].astype(int); out['pred_missing_mask'] = ms
    os.makedirs('working', exist_ok=True); out.sort_values('row_id').to_csv('working/submission.csv', index=False)
    print("WROTE FALLBACK submission", flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("main() failed -> fallback:", e, flush=True)
        fallback()
