import os, time, math, argparse, copy, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GMAX = 10
INF = 1e15

# ---------------- data ----------------
def load_cache(split):
    img = np.load(f'_cache_{split}_img.npy'); vols = [str(v) for v in np.load(f'_cache_{split}_vols.npy')]
    return img, vols

def build_targets(csv, vols):
    df = pd.read_csv(csv, dtype={'volume_id': str}); df['volume_id'] = df['volume_id'].apply(lambda x: str(int(x)))
    pos = np.zeros((len(vols), 32), np.int64); rank = np.zeros((len(vols), 32), np.int64); mask = np.zeros((len(vols), 48), np.int64)
    for vi, v in enumerate(vols):
        sub = df[df['volume_id'] == v].sort_values('presented_index')
        m = np.array([int(c) for c in sub['true_missing_mask'].iloc[0][1:]]); mask[vi] = m
        present = np.where(m == 0)[0]; r = sub['true_rank'].to_numpy(); rank[vi] = r
        L = len(present); idx = np.round(r*(L-1)/31.0).astype(int).clip(0, L-1); pos[vi] = present[idx]
    return pos, rank, mask

# ---------------- model ----------------
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
        s.l1 = nn.Sequential(Res(48, 64, 2), Res(64, 64, 1))
        s.l2 = nn.Sequential(Res(64, 128, 2), Res(128, 128, 1))
        s.l3 = nn.Sequential(Res(128, 256, 2), Res(256, 256, 1))
        s.l4 = nn.Sequential(Res(256, d, 2), Res(d, d, 1))
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
        s.trail_head = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, GMAX))
        s.d = d
    def embed(s, x, flip_tta=False):
        B, T = x.shape[:2]; xf = x.reshape(B*T, *x.shape[2:])
        e = s.enc(xf)
        if flip_tta: e = 0.5*(e+s.enc(torch.flip(xf, dims=[3])))
        return s.norm(s.tx(s.proj(e.reshape(B, T, -1))))
    def pos(s, h): return s.pos_head(h).squeeze(-1)
    def cls(s, h): return s.cls_head(h)          # (B,32,48)
    def gaps(s, h, order):
        c = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, s.d))
        g, _ = s.gru(c); a, b = g[:, :-1], g[:, 1:]
        feat = torch.cat([a, b, a-b, (a-b).abs()], -1)
        gl = s.gap_head(feat); ctx = g.mean(1)
        return gl, s.lead_head(torch.cat([g[:, 0], ctx], -1)), s.trail_head(torch.cat([g[:, -1], ctx], -1))

# ---------------- aug ----------------
def augment(x):
    B, T, _, R, _ = x.shape; n = B*T; x = x.reshape(n, 1, R, R)
    flip = torch.rand(n, device=x.device) < 0.5; x[flip] = torch.flip(x[flip], dims=[3])
    ang = (torch.rand(n, device=x.device)*2-1)*(10*math.pi/180)
    tx = (torch.rand(n, device=x.device)*2-1)*0.05; ty = (torch.rand(n, device=x.device)*2-1)*0.05
    sc = 1+(torch.rand(n, device=x.device)*2-1)*0.06; cos, sin = torch.cos(ang)*sc, torch.sin(ang)*sc
    th = torch.zeros(n, 2, 3, device=x.device); th[:, 0, 0]=cos; th[:, 0, 1]=-sin; th[:, 0, 2]=tx; th[:, 1, 0]=sin; th[:, 1, 1]=cos; th[:, 1, 2]=ty
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False, padding_mode='zeros')
    gm = (0.8+0.45*torch.rand(n, 1, 1, 1, device=x.device)); x = x.clamp(1e-4, 1).pow(gm)
    x = x*(0.85+0.3*torch.rand(n, 1, 1, 1, device=x.device)) + (torch.rand(n, 1, 1, 1, device=x.device)*2-1)*0.05
    return (x+torch.randn_like(x)*0.03).clamp(0, 1).reshape(B, T, 1, R, R)

def pairwise_rank_loss(sc, tgt):
    d = sc.unsqueeze(2)-sc.unsqueeze(1); t = tgt.unsqueeze(2)-tgt.unsqueeze(1); m = (t != 0).float()
    return (F.softplus(-torch.sign(t)*d)*m).sum()/m.sum().clamp(min=1)

# ---------------- decode ----------------
def macro_f1(p, t): return f1_score(t, p, average='macro', labels=[0, 1], zero_division=0)
def logsm1(x): x = x-x.max(); e = np.exp(x); return np.log(e/e.sum()+1e-12)
def logsm2(x): x = x-x.max(1, keepdims=True); e = np.exp(x); return np.log(e/e.sum(1, keepdims=True)+1e-12)

_A = 48; _ii = np.arange(_A); _D = (_ii[None, :]-_ii[:, None]-1).clip(0, _A-1); _UP = _ii[None, :] > _ii[:, None]
def _tailvec(arr):  # extend (GMAX,) to (48,) with linear tail
    out = np.empty(_A)
    out[:GMAX] = arr[:GMAX]
    for g in range(GMAX, _A): out[g] = arr[GMAX-1]-(g-GMAX+1)*2.0
    return out

def unified_dp(cls_lp, gli, leadlp, traillp, pri, w_cls, w_gap, w_iprior, w_lead, w_lprior, w_occ, lp_present):
    # cls_lp:(32,48) ORDERED node logprob ; gli:(31,GMAX); leadlp/traillp:(GMAX,)
    A = _A
    gc = -w_gap*np.stack([_tailvec(gli[v]) for v in range(31)]) - w_iprior*_tailvec(pri['internal'])[None, :]  # (31,48)
    node = -w_cls*cls_lp - w_occ*lp_present[None, :]
    leadv = _tailvec(leadlp); trailv = _tailvec(traillp); lpri = _tailvec(pri['lead']); tpri = _tailvec(pri['trail'])
    dp = node[0] - w_lead*leadv - w_lprior*lpri
    back = np.zeros((32, A), int)
    for v in range(1, 32):
        gmat = gc[v-1][_D].copy(); gmat[~_UP] = INF      # gmat[a',a] = gapcost(a-a'-1)
        M = dp[:, None] + gmat
        prev = np.argmin(M, axis=0); dp = M[prev, _ii] + node[v]
        if v == 31:
            dp = dp - w_lead*trailv[47-_ii] - w_lprior*tpri[47-_ii]
        back[v] = prev
    a = int(np.argmin(dp)); slots = []
    for v in range(31, -1, -1): slots.append(a); a = back[v, a]
    m = np.ones(48, int); m[slots] = 0; return m

def decode_all(POS, CLS, GL, LE, TR, pri, lp_present, W):
    out = []
    for i in range(len(POS)):
        order = np.argsort(POS[i])
        cls_lp = logsm2(CLS[i])[order]                       # ORDERED node potentials
        gli = logsm2(GL[i]); lead = logsm1(LE[i]); trail = logsm1(TR[i])
        m = unified_dp(cls_lp, gli, lead, trail, pri, **W, lp_present=lp_present)
        out.append(m)
    return out

# ---------------- train ----------------
@torch.no_grad()
def predict(model, img, idx, R, flip_tta=True, bs=8):
    model.eval(); POS=[]; CLS=[]; GL=[]; LE=[]; TR=[]
    for s in range(0, len(idx), bs):
        ii = idx[s:s+bs]; x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if R != 192:
            b, t = x.shape[:2]; x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(R, R), mode='bilinear', align_corners=False).reshape(b, t, 1, R, R)
        h = model.embed(x, flip_tta=flip_tta); pos = model.pos(h).cpu().numpy(); cls = model.cls(h).cpu().numpy()
        order = np.argsort(pos, axis=1); gl, le, tr = model.gaps(h, torch.from_numpy(order).to(DEV))
        POS.append(pos); CLS.append(cls); GL.append(gl.cpu().numpy()); LE.append(le.cpu().numpy()); TR.append(tr.cpu().numpy())
    return np.concatenate(POS), np.concatenate(CLS), np.concatenate(GL), np.concatenate(LE), np.concatenate(TR)

def eval_full(preds, rank, mask, pri, lp_present, Wgrid):
    POS, CLS, GL, LE, TR = preds
    taus = [max(0, kendalltau(np.argsort(np.argsort(POS[i])), rank[i]).statistic)**2 for i in range(len(POS))]
    rank2 = float(np.mean(taus)); best = (-1, None, None)
    for W in Wgrid:
        ms = decode_all(POS, CLS, GL, LE, TR, pri, lp_present, W)
        f1sq = np.mean([macro_f1(ms[i], mask[i])**2 for i in range(len(ms))])
        mf = np.mean([macro_f1(ms[i], mask[i]) for i in range(len(ms))])
        if f1sq > best[0]: best = (f1sq, W, mf)
    return rank2, best

def make_soft_targets(pos, sigma=1.2):
    j = np.arange(48)[None, None, :]; t = pos[:, :, None]
    w = np.exp(-0.5*((j-t)/sigma)**2); return (w/w.sum(-1, keepdims=True)).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--R', type=int, default=160); ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--nval', type=int, default=96); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ema', type=float, default=0.998)
    args = ap.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)

    img, vols = load_cache('train'); pos, rank, mask = build_targets('train.csv', vols); N = len(vols)
    perm = np.random.RandomState(args.seed).permutation(N); val_idx = perm[:args.nval]; trn_idx = perm[args.nval:]
    tm = mask[trn_idx]
    lead_c = np.zeros(GMAX); trail_c = np.zeros(GMAX); int_c = np.zeros(GMAX)
    for r in tm:
        p = np.where(r == 0)[0]; lead_c[min(p[0], GMAX-1)] += 1; trail_c[min(47-p[-1], GMAX-1)] += 1
        for g in np.diff(p)-1: int_c[min(max(g, 0), GMAX-1)] += 1
    pri = {'lead': np.log(lead_c/lead_c.sum()+1e-9), 'trail': np.log(trail_c/trail_c.sum()+1e-9), 'internal': np.log(int_c/int_c.sum()+1e-9)}
    int_w = np.clip(int_c.sum()/(GMAX*(int_c+1)), 0.2, 8.0); cls_w = torch.tensor(int_w, dtype=torch.float32, device=DEV)
    lp_present = np.log((1-tm.mean(0)).clip(.02, .98)/tm.mean(0).clip(.02, .98))
    soft = torch.from_numpy(make_soft_targets(pos)).to(DEV)

    model = Net().to(DEV); ema = copy.deepcopy(model); [p.requires_grad_(False) for p in ema.parameters()]
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M train={len(trn_idx)} val={len(val_idx)} R={args.R}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs*(len(trn_idx)//args.bs+1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.08)
    scaler = torch.cuda.amp.GradScaler(); post = torch.from_numpy(pos).to(DEV).float(); rankt = torch.from_numpy(rank).to(DEV)

    def get_batch(ii):
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if args.R != 192:
            b, t = x.shape[:2]; x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(args.R, args.R), mode='bilinear', align_corners=False).reshape(b, t, 1, args.R, args.R)
        return x

    Wgrid = []
    for w_cls in [0.5, 1.0, 2.0]:
        for w_gap in [1.0, 2.0]:
            for w_lead in [0.0, 1.0]:
                for w_lprior in [0.5, 1.5]:
                    Wgrid.append(dict(w_cls=w_cls, w_gap=w_gap, w_iprior=0.3, w_lead=w_lead, w_lprior=w_lprior, w_occ=0.3))

    t0 = time.time(); gstep = 0
    for ep in range(args.epochs):
        model.train(); np.random.shuffle(trn_idx); tl = 0; nb = 0
        for s in range(0, len(trn_idx), args.bs):
            ii = trn_idx[s:s+args.bs]; x = augment(get_batch(ii)); tgt = post[ii]; rk = rankt[ii]
            order = torch.argsort(rk, dim=1)
            with torch.cuda.amp.autocast():
                h = model.embed(x); pscore = model.pos(h); clsl = model.cls(h)
                reg = F.smooth_l1_loss(pscore, tgt/47.0); rkl = pairwise_rank_loss(pscore, tgt)
                cls_loss = -(soft[ii]*F.log_softmax(clsl, -1)).sum(-1).mean()
                gl, lead, trail = model.gaps(h, order); pos_ord = torch.gather(tgt, 1, order)
                gtar = (pos_ord[:, 1:]-pos_ord[:, :-1]-1).clamp(0, GMAX-1).long()
                ltar = pos_ord[:, 0].clamp(0, GMAX-1).long(); ttar = (47-pos_ord[:, 31]).clamp(0, GMAX-1).long()
                gap_loss = F.cross_entropy(gl.reshape(-1, GMAX), gtar.reshape(-1), weight=cls_w, label_smoothing=0.05)
                edge_loss = F.cross_entropy(lead, ltar, label_smoothing=0.05)+F.cross_entropy(trail, ttar, label_smoothing=0.05)
                loss = 1.0*reg + 1.0*rkl + 1.0*cls_loss + 0.3*gap_loss + 0.1*edge_loss
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            gstep += 1; dec = min(args.ema, (1+gstep)/(10+gstep))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(dec).add_(pm, alpha=1-dec)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            tl += loss.item(); nb += 1
        if ep % 5 == 4 or ep >= args.epochs-3:
            preds = predict(model, img, val_idx, args.R); rank2, best = eval_full(preds, rank[val_idx], mask[val_idx], pri, lp_present, Wgrid)
            print(f"ep{ep+1} loss{tl/nb:.3f} | RAW rank2={rank2:.4f} mask2={best[0]:.4f} mf1={best[2]:.4f} S={0.6*rank2+0.4*best[0]:.4f} W={best[1]} | {time.time()-t0:.0f}s")
    # final: pick better of raw/ema
    fin = {}
    for nm, mdl in [('raw', model), ('ema', ema)]:
        preds = predict(mdl, img, val_idx, args.R); r2, bb = eval_full(preds, rank[val_idx], mask[val_idx], pri, lp_present, Wgrid)
        fin[nm] = (0.6*r2+0.4*bb[0], r2, bb); print(f"FINAL {nm}: rank2={r2:.4f} mask2={bb[0]:.4f} mf1={bb[2]:.4f} S={fin[nm][0]:.4f} W={bb[1]}")
    bestnm = max(fin, key=lambda k: fin[k][0]); mdl = model if bestnm == 'raw' else ema
    torch.save({'model': mdl.state_dict(), 'prior': pri, 'lp_present': lp_present, 'R': args.R, 'W': fin[bestnm][2][1]}, f'_m3_seed{args.seed}.pt')
    print(f"saved {bestnm} S={fin[bestnm][0]:.4f} W={fin[bestnm][2][1]}")

if __name__ == '__main__':
    main()
