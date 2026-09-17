import os, time, math, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GMAX = 10  # gap classes 0..9

# ---------------- data ----------------
def load_cache(split):
    img = np.load(f'_cache_{split}_img.npy')
    vols = [str(v) for v in np.load(f'_cache_{split}_vols.npy')]
    return img, vols

def build_targets(csv, vols):
    df = pd.read_csv(csv, dtype={'volume_id': str})
    df['volume_id'] = df['volume_id'].apply(lambda x: str(int(x)))
    pos = np.zeros((len(vols), 32), np.int64)
    rank = np.zeros((len(vols), 32), np.int64)
    mask = np.zeros((len(vols), 48), np.int64)
    for vi, v in enumerate(vols):
        sub = df[df['volume_id'] == v].sort_values('presented_index')
        m = np.array([int(c) for c in sub['true_missing_mask'].iloc[0][1:]])
        mask[vi] = m
        present = np.where(m == 0)[0]
        r = sub['true_rank'].to_numpy(); rank[vi] = r
        L = len(present)
        idx = np.round(r * (L - 1) / 31.0).astype(int).clip(0, L - 1)
        pos[vi] = present[idx]
    return pos, rank, mask

# ---------------- model ----------------
class Res(nn.Module):
    def __init__(s, ci, co, stride):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(co)
        s.sc = nn.Sequential() if (stride == 1 and ci == co) else nn.Sequential(
            nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))
    def forward(s, x):
        y = F.relu(s.b1(s.c1(x)), True); y = s.b2(s.c2(y))
        return F.relu(y + s.sc(x), True)

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
        s.enc = Encoder(denc)
        s.proj = nn.Linear(denc, d)
        enc = nn.TransformerEncoderLayer(d, heads, d*2, 0.1, 'gelu', batch_first=True, norm_first=True)
        s.tx = nn.TransformerEncoder(enc, layers)
        s.norm = nn.LayerNorm(d)
        s.pos_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        s.gru = nn.GRU(d, d//2, batch_first=True, bidirectional=True)
        s.gap_head = nn.Sequential(nn.Linear(d*4, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, GMAX))
        s.lead_head = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, GMAX))
        s.trail_head = nn.Sequential(nn.Linear(d*2, d), nn.GELU(), nn.Linear(d, GMAX))
        s.d = d

    def embed(s, x, flip_tta=False):
        B, T = x.shape[:2]
        xf = x.reshape(B*T, *x.shape[2:])
        e = s.enc(xf)
        if flip_tta:
            e = 0.5*(e + s.enc(torch.flip(xf, dims=[3])))
        e = e.reshape(B, T, -1)
        h = s.norm(s.tx(s.proj(e)))      # (B,32,d) contextual
        return h

    def pos(s, h):
        return s.pos_head(h).squeeze(-1)  # (B,32)

    def gaps(s, h, order):
        # order: (B,32) long indices giving caudal->cranial order
        c = torch.gather(h, 1, order.unsqueeze(-1).expand(-1, -1, s.d))  # (B,32,d) ordered
        g, _ = s.gru(c)                                                  # (B,32,d)
        a, b = g[:, :-1], g[:, 1:]                                       # (B,31,d)
        feat = torch.cat([a, b, a-b, (a-b).abs()], -1)                   # (B,31,4d)
        gap_logits = s.gap_head(feat)                                    # (B,31,GMAX)
        ctx = g.mean(1)                                                  # (B,d)
        lead = s.lead_head(torch.cat([g[:, 0], ctx], -1))               # (B,GMAX)
        trail = s.trail_head(torch.cat([g[:, -1], ctx], -1))            # (B,GMAX)
        return gap_logits, lead, trail

# ---------------- aug ----------------
def augment(x):
    B, T, _, R, _ = x.shape; n = B*T
    x = x.reshape(n, 1, R, R)
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = torch.flip(x[flip], dims=[3])
    ang = (torch.rand(n, device=x.device)*2-1)*(10*math.pi/180)
    tx = (torch.rand(n, device=x.device)*2-1)*0.05
    ty = (torch.rand(n, device=x.device)*2-1)*0.05
    sc = 1+(torch.rand(n, device=x.device)*2-1)*0.06
    cos, sin = torch.cos(ang)*sc, torch.sin(ang)*sc
    th = torch.zeros(n, 2, 3, device=x.device)
    th[:, 0, 0]=cos; th[:, 0, 1]=-sin; th[:, 0, 2]=tx
    th[:, 1, 0]=sin; th[:, 1, 1]=cos; th[:, 1, 2]=ty
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
    g = (0.8+0.45*torch.rand(n, 1, 1, 1, device=x.device))
    x = x.clamp(1e-4, 1).pow(g)
    x = x*(0.85+0.3*torch.rand(n, 1, 1, 1, device=x.device)) + (torch.rand(n, 1, 1, 1, device=x.device)*2-1)*0.05
    x = x + torch.randn_like(x)*0.03
    return x.clamp(0, 1).reshape(B, T, 1, R, R)

def pairwise_rank_loss(sc, target):
    d = sc.unsqueeze(2)-sc.unsqueeze(1); t = target.unsqueeze(2)-target.unsqueeze(1)
    m = (t != 0).float()
    return (F.softplus(-torch.sign(t)*d)*m).sum()/m.sum().clamp(min=1)

# ---------------- decode ----------------
def macro_f1_mask(pred48, true48):
    return f1_score(true48, pred48, average='macro', labels=[0, 1], zero_division=0)

def count_dp(slot_logp):
    # slot_logp: list of 33 arrays (GMAX,) ; maximize sum subject to total gaps == 16
    S = len(slot_logp); B = 16
    NEG = -1e18
    f = np.full(B+1, NEG); f[0] = 0.0
    bk = np.full((S, B+1), -1, int)
    for s in range(S):
        nf = np.full(B+1, NEG); lp = slot_logp[s]
        for b in range(B+1):
            best = NEG; bg = 0
            gmax = min(GMAX-1, b)
            for g in range(gmax+1):
                v = f[b-g] + lp[g]
                if v > best: best = v; bg = g
            nf[b] = best; bk[s, b] = bg
        f = nf
    gaps = np.zeros(S, int); b = B
    for s in range(S-1, -1, -1):
        gaps[s] = bk[s, b]; b -= gaps[s]
    return gaps  # gaps[0]=lead, gaps[1..31]=internal, gaps[32]=trail

def build_mask_from_gaps(gaps):
    m = np.ones(48, int); idx = 0
    idx += gaps[0]
    for k in range(32):
        if idx < 48: m[idx] = 0
        idx += 1
        if k < 31: idx += gaps[1+k]
    # gaps[32] trailing already accounted by remaining
    return m

def decode_volume(pos_k, gap_logp_internal, lead_logp, trail_logp, prior, lam):
    # pos_k (32,), gap_logp_internal (31,GMAX) ordered, lead/trail (GMAX,)
    order = np.argsort(pos_k)
    # build slot logp with prior blend
    li, ll, lt = prior['internal'], prior['lead'], prior['trail']
    slots = []
    slots.append((1-lam)*lead_logp + lam*ll)
    for k in range(31):
        slots.append((1-lam)*gap_logp_internal[k] + lam*li)
    slots.append((1-lam)*trail_logp + lam*lt)
    gaps = count_dp(slots)
    m = build_mask_from_gaps(gaps)
    # repair: ensure exactly 16 ones
    s = m.sum()
    if s != 16:
        # shouldn't happen (DP forces 16) but guard
        if s > 16:
            ones = np.where(m == 1)[0]
            m[ones[16:]] = 0
        else:
            zeros = np.where(m == 0)[0]
            m[zeros[:16-s]] = 1
    return order, m

def logsm(x):
    x = x - x.max(); e = np.exp(x); return np.log(e/e.sum()+1e-12)

# ---------------- eval ----------------
@torch.no_grad()
def predict(model, img, idx, R, flip_tta=True, bs=8):
    model.eval(); P=[]; GI=[]; LE=[]; TR=[]
    for s in range(0, len(idx), bs):
        ii = idx[s:s+bs]
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if R != 192:
            b, t = x.shape[:2]
            x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(R, R), mode='bilinear', align_corners=False).reshape(b, t, 1, R, R)
        h = model.embed(x, flip_tta=flip_tta)
        pos = model.pos(h).cpu().numpy()
        order = np.argsort(pos, axis=1)
        ot = torch.from_numpy(order).to(DEV)
        gl, le, tr = model.gaps(h, ot)
        P.append(pos); GI.append(gl.cpu().numpy()); LE.append(le.cpu().numpy()); TR.append(tr.cpu().numpy())
    return np.concatenate(P), np.concatenate(GI), np.concatenate(LE), np.concatenate(TR)

def eval_metric(pos, gi, le, tr, rank, mask, prior, lam, prior_mask):
    taus, f1sq, f1s, recalls = [], [], [], []
    n = len(pos)
    for i in range(n):
        pr = np.argsort(np.argsort(pos[i]))
        tau = kendalltau(pr, rank[i]).statistic
        taus.append(max(0, tau)**2 if tau == tau else 0)
        order = np.argsort(pos[i])
        gli = np.stack([logsm(gi[i][k]) for k in range(31)])
        lel = logsm(le[i]); trl = logsm(tr[i])
        _, m = decode_volume(pos[i], gli, lel, trl, prior, lam)
        # guardrail handled by caller via lam grid; here just gap-DP
        f = macro_f1_mask(m, mask[i]); f1s.append(f); f1sq.append(f**2)
        true_present = np.where(mask[i] == 0)[0]
        # gap>=1 recall: among true internal gaps>=1, how many predicted>=1 (rough)
    rank_sq = float(np.mean(taus)); mask_sq = float(np.mean(f1sq))
    return 0.6*rank_sq+0.4*mask_sq, rank_sq, mask_sq, float(np.mean(f1s))

# ---------------- train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--R', type=int, default=128)
    ap.add_argument('--bs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--nval', type=int, default=96)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    img, vols = load_cache('train')
    pos, rank, mask = build_targets('train.csv', vols)
    N = len(vols)
    rs = np.random.RandomState(args.seed); perm = rs.permutation(N)
    val_idx = perm[:args.nval]; trn_idx = perm[args.nval:]

    # priors from TRAIN only
    tm = mask[trn_idx]
    present_list = [np.where(r == 0)[0] for r in tm]
    lead_c = np.zeros(GMAX); trail_c = np.zeros(GMAX); int_c = np.zeros(GMAX)
    for p in present_list:
        lead_c[min(p[0], GMAX-1)] += 1
        trail_c[min(47-p[-1], GMAX-1)] += 1
        for g in np.diff(p)-1:
            int_c[min(max(g, 0), GMAX-1)] += 1
    prior = {'lead': np.log(lead_c/lead_c.sum()+1e-9),
             'trail': np.log(trail_c/trail_c.sum()+1e-9),
             'internal': np.log(int_c/int_c.sum()+1e-9)}
    int_w = (int_c.sum()/(GMAX*(int_c+1))); int_w = np.clip(int_w, 0.2, 8.0)
    cls_w = torch.tensor(int_w, dtype=torch.float32, device=DEV)
    # constant prior mask (top16 marginal) for guardrail
    marg = tm.mean(0); prior_mask = np.zeros(48, int); prior_mask[np.argsort(-marg)[:16]] = 1

    model = Net().to(DEV)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M  train={len(trn_idx)} val={len(val_idx)} R={args.R}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs*(len(trn_idx)//args.bs+1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.08)
    scaler = torch.cuda.amp.GradScaler()
    post = torch.from_numpy(pos).to(DEV).float()
    rankt = torch.from_numpy(rank).to(DEV)

    def get_batch(ii):
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if args.R != 192:
            b, t = x.shape[:2]
            x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(args.R, args.R), mode='bilinear', align_corners=False).reshape(b, t, 1, args.R, args.R)
        return x

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); rs.shuffle(trn_idx); tl = gl_ = 0; nb = 0
        for s in range(0, len(trn_idx), args.bs):
            ii = trn_idx[s:s+args.bs]
            x = augment(get_batch(ii))
            tgt = post[ii]; rk = rankt[ii]
            order = torch.argsort(rk, dim=1)         # caudal->cranial (teacher forcing)
            with torch.cuda.amp.autocast():
                h = model.embed(x, flip_tta=False)
                pscore = model.pos(h)
                reg = F.smooth_l1_loss(pscore, tgt/47.0)
                rkl = pairwise_rank_loss(pscore, tgt)
                gap_logits, lead, trail = model.gaps(h, order)
                pos_ord = torch.gather(tgt, 1, order)            # (B,32)
                gtar = (pos_ord[:, 1:]-pos_ord[:, :-1]-1).clamp(0, GMAX-1).long()  # (B,31)
                ltar = pos_ord[:, 0].clamp(0, GMAX-1).long()
                ttar = (47-pos_ord[:, 31]).clamp(0, GMAX-1).long()
                gap_loss = F.cross_entropy(gap_logits.reshape(-1, GMAX), gtar.reshape(-1), weight=cls_w, label_smoothing=0.05)
                edge_loss = F.cross_entropy(lead, ltar, label_smoothing=0.05)+F.cross_entropy(trail, ttar, label_smoothing=0.05)
                loss = 1.0*reg + 1.0*rkl + 0.3*gap_loss + 0.1*edge_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); gl_ += gap_loss.item(); nb += 1
        if ep % 5 == 4 or ep == args.epochs-1:
            P, GI, LE, TR = predict(model, img, val_idx, args.R)
            best = (-1, 0)
            for lam in [0.0, 0.1, 0.15, 0.2, 0.3, 0.5]:
                S, rsq, msq, mf1 = eval_metric(P, GI, LE, TR, rank[val_idx], mask[val_idx], prior, lam, prior_mask)
                if msq > best[0]: best = (msq, lam, S, rsq, msq, mf1)
            print(f"ep{ep+1} loss{tl/nb:.3f} gap{gl_/nb:.3f} | val S={best[2]:.4f} rank2={best[3]:.4f} mask2={best[4]:.4f} mf1={best[5]:.4f} lam={best[1]} | {time.time()-t0:.0f}s")
    torch.save({'model': model.state_dict(), 'prior': prior, 'prior_mask': prior_mask, 'R': args.R}, f'_m2_seed{args.seed}.pt')
    print("saved")
    diagnose(model, img, val_idx, pos, rank, mask, prior, args.R)


@torch.no_grad()
def diagnose(model, img, idx, pos, rank, mask, prior, R):
    from sklearn.metrics import roc_auc_score
    model.eval()
    pos_sp, emb_d, true_g, gap_pred = [], [], [], []
    rank_taus = []
    for s in range(0, len(idx), 8):
        ii = idx[s:s+8]
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if R != 192:
            b, t = x.shape[:2]
            x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(R, R), mode='bilinear', align_corners=False).reshape(b, t, 1, R, R)
        h = model.embed(x, flip_tta=True)
        pscore = model.pos(h).cpu().numpy()
        # TRUE order
        torder = np.argsort(rank[ii], axis=1)
        ot = torch.from_numpy(torder).to(DEV)
        gl, le, tr = model.gaps(h, ot)
        gl = gl.cpu().numpy()
        hc = h.cpu().numpy()
        for bi, vi in enumerate(ii):
            pr = np.argsort(np.argsort(pscore[bi]))
            tau = kendalltau(pr, rank[vi]).statistic
            rank_taus.append(tau)
            o = torder[bi]
            pos_ord = pos[vi][o]; psc_ord = pscore[bi][o]; h_ord = hc[bi][o]
            for k in range(31):
                g = int(min(max(pos_ord[k+1]-pos_ord[k]-1, 0), 9))
                true_g.append(g)
                pos_sp.append(psc_ord[k+1]-psc_ord[k])
                emb_d.append(np.linalg.norm(h_ord[k+1]-h_ord[k]))
                gap_pred.append(int(np.argmax(gl[bi][k])))
    true_g = np.array(true_g); pos_sp = np.array(pos_sp); emb_d = np.array(emb_d); gap_pred = np.array(gap_pred)
    y = (true_g >= 1).astype(int)
    print(f"\n=== DIAGNOSTIC === mean|tau|={np.mean(np.abs(rank_taus)):.4f}")
    print(f"gap>=1 base rate {y.mean():.3f}  (n pairs {len(y)})")
    try:
        print(f"AUC gap>=1 from pos-spacing: {roc_auc_score(y, pos_sp):.4f}")
        print(f"AUC gap>=1 from emb-dist:    {roc_auc_score(y, emb_d):.4f}")
    except Exception as e:
        print("auc err", e)
    print(f"gap-head argmax vs true gap (recall gap>=1): pred>=1 & true>=1 = {((gap_pred>=1)&(y==1)).sum()}/{ (y==1).sum() } "
          f"= {((gap_pred>=1)&(y==1)).sum()/max((y==1).sum(),1):.3f}; "
          f"precision = {((gap_pred>=1)&(y==1)).sum()/max((gap_pred>=1).sum(),1):.3f}")
    print("true gap hist:", np.bincount(true_g, minlength=5)[:5], " gap-head pred hist:", np.bincount(gap_pred, minlength=5)[:5])
    # pos-spacing mean by true gap
    for g in range(4):
        sel = true_g == g
        if sel.sum() > 0:
            print(f"  true gap={g}: pos-spacing mean={pos_sp[sel].mean():.3f} std={pos_sp[sel].std():.3f}  emb-dist mean={emb_d[sel].mean():.3f}")

if __name__ == '__main__':
    main()
