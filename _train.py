import os, time, math, argparse, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import kendalltau
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- data ----------------
def load_cache(split):
    img = np.load(f'_cache_{split}_img.npy')           # (N,32,192,192) uint8
    vols = np.load(f'_cache_{split}_vols.npy')
    return img, [str(v) for v in vols]

def build_targets(csv, vols):
    df = pd.read_csv(csv, dtype={'volume_id': str})
    df['volume_id'] = df['volume_id'].apply(lambda x: str(int(x)))
    pos = np.zeros((len(vols), 32), np.int64)          # absolute window position 0..47 per presented slice
    rank = np.zeros((len(vols), 32), np.int64)
    mask = np.zeros((len(vols), 48), np.int64)
    for vi, v in enumerate(vols):
        sub = df[df['volume_id'] == v].sort_values('presented_index')
        m = np.array([int(c) for c in sub['true_missing_mask'].iloc[0][1:]])
        mask[vi] = m
        present = np.where(m == 0)[0]                   # ~32 positions, caudal->cranial (15-17 ones => 31-33)
        r = sub['true_rank'].to_numpy()                 # rank of each presented slice
        rank[vi] = r
        # monotone map rank 0..31 -> a present position; exact when |present|==32, graceful otherwise
        L = len(present)
        idx = np.round(r * (L - 1) / 31.0).astype(int).clip(0, L - 1)
        pos[vi] = present[idx]                          # presented slice -> its absolute position
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
        y = F.relu(s.b1(s.c1(x)), inplace=True); y = s.b2(s.c2(y))
        return F.relu(y + s.sc(x), inplace=True)

class Encoder(nn.Module):
    def __init__(s, d=256):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))  # /2
        s.l1 = Res(32, 64, 2)    # /4
        s.l2 = Res(64, 96, 2)    # /8
        s.l3 = Res(96, 160, 2)   # /16
        s.l4 = Res(160, d, 2)    # /32
        s.d = d
    def forward(s, x):
        x = s.stem(x); x = s.l1(x); x = s.l2(x); x = s.l3(x); x = s.l4(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B,d)

class SetTx(nn.Module):
    def __init__(s, d=256, layers=3, heads=4):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d, heads, d*2, 0.1, 'gelu', batch_first=True, norm_first=True)
        s.tx = nn.TransformerEncoder(enc, layers)
    def forward(s, x):  # (B,32,d)
        return s.tx(x)

class Net(nn.Module):
    def __init__(s, d=256):
        super().__init__()
        s.enc = Encoder(d); s.tx = SetTx(d)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(s, x):           # x (B,32,1,R,R)
        B, T = x.shape[:2]
        e = s.enc(x.reshape(B*T, *x.shape[2:])).reshape(B, T, -1)
        h = s.tx(e)
        h = s.norm(h)
        sc = s.head(h).squeeze(-1)   # (B,32) predicted position (normalized space)
        return sc, e

# ---------------- aug (GPU) ----------------
def augment(x, train=True):
    # x: (B,32,1,R,R) float in [0,1]
    B, T, _, R, _ = x.shape
    n = B*T
    x = x.reshape(n, 1, R, R)
    if train:
        flip = torch.rand(n, device=x.device) < 0.5
        x[flip] = torch.flip(x[flip], dims=[3])
        ang = (torch.rand(n, device=x.device)*2-1) * (8*math.pi/180)
        tx = (torch.rand(n, device=x.device)*2-1) * 0.05
        ty = (torch.rand(n, device=x.device)*2-1) * 0.05
        sc = 1 + (torch.rand(n, device=x.device)*2-1) * 0.06
        cos, sin = torch.cos(ang)*sc, torch.sin(ang)*sc
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
        theta[:, 1, 0] = sin; theta[:, 1, 1] = cos; theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')
        g = (0.8 + 0.4*torch.rand(n, 1, 1, 1, device=x.device))   # gamma
        x = x.clamp(1e-4, 1).pow(g)
        x = x * (0.85 + 0.3*torch.rand(n, 1, 1, 1, device=x.device)) + (torch.rand(n,1,1,1,device=x.device)*2-1)*0.05
        x = x + torch.randn_like(x) * 0.03
    x = x.clamp(0, 1)
    return x.reshape(B, T, 1, R, R)

# ---------------- losses ----------------
def pairwise_rank_loss(sc, target):
    # sc,target: (B,32). logistic ranking over all pairs
    d = sc.unsqueeze(2) - sc.unsqueeze(1)          # (B,32,32) pred diff
    t = target.unsqueeze(2) - target.unsqueeze(1)  # true diff
    sign = torch.sign(t)
    mask = (t != 0).float()
    loss = F.softplus(-sign * d) * mask
    return loss.sum() / mask.sum().clamp(min=1)

# ---------------- metric / decode ----------------
def macro_f1_mask(pred48, true48):
    return f1_score(true48, pred48, average='macro', labels=[0, 1], zero_division=0)

def decode_mask(q, lp_present, beta, scale):
    # q: (32,) predicted positions (sorted ascending). DP assign to 0..47 monotone.
    q = q * scale
    cost = (q[:, None] - np.arange(48)[None, :])**2 - beta * lp_present[None, :]  # (32,48)
    INF = 1e18
    dp = np.full((32, 48), INF); bk = np.full((32, 48), -1, int)
    dp[0] = cost[0]
    for k in range(1, 32):
        run = INF; argrun = -1
        best = np.full(48, INF); ba = np.full(48, -1, int)
        for j in range(48):
            if j-1 >= 0 and dp[k-1, j-1] < run:
                run = dp[k-1, j-1]; argrun = j-1
            if run < INF:
                best[j] = run + cost[k, j]; ba[j] = argrun
        dp[k] = best; bk[k] = ba
    j = int(np.argmin(dp[31])); present = []
    for k in range(31, -1, -1):
        present.append(j); j = bk[k, j]
    present = present[::-1]
    m = np.ones(48, int); m[present] = 0
    return m

def score_split(model, img, pos, rank, mask, idx, lp_present, beta=2.0, scale=1.0, tta=True, bs=8):
    model.eval(); R = img.shape[-1]
    taus, f1s, f1sq = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            ii = idx[s:s+bs]
            x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)  # (b,32,1,R,R)
            sc, _ = model(x)
            if tta:
                sc2, _ = model(torch.flip(x, dims=[4]))
                sc = (sc + sc2) / 2
            sc = sc.cpu().numpy()
            for bi, vi in enumerate(ii):
                s_k = sc[bi]
                pr = np.argsort(np.argsort(s_k))           # pred rank 0..31
                tau = kendalltau(pr, rank[vi]).statistic
                taus.append(max(0, tau)**2 if tau == tau else 0)
                order = np.argsort(s_k)
                q = s_k[order]
                q = (q - q.min()) / (q.max() - q.min() + 1e-6) * 47   # normalize span to 0..47
                m = decode_mask(q, lp_present, beta, scale)
                f = macro_f1_mask(m, mask[vi])
                f1s.append(f); f1sq.append(f**2)
    rank_sq = float(np.mean(taus)); mask_sq = float(np.mean(f1sq))
    S = 0.6*rank_sq + 0.4*mask_sq
    return S, rank_sq, mask_sq, float(np.mean(f1s))

# ---------------- train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--R', type=int, default=128)
    ap.add_argument('--bs', type=int, default=12)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--nval', type=int, default=80)
    args = ap.parse_args()

    img, vols = load_cache('train')
    pos, rank, mask = build_targets('train.csv', vols)
    N = len(vols)
    rs = np.random.RandomState(0); perm = rs.permutation(N)
    val_idx = perm[:args.nval]; trn_idx = perm[args.nval:]
    lp_present = np.log((1 - mask[trn_idx].mean(0)).clip(0.02, 0.98) / mask[trn_idx].mean(0).clip(0.02, 0.98))
    R = args.R
    # resize cache once on CPU via torch interpolate per-batch (keep uint8 in RAM, resize on GPU)
    print(f"N={N} train={len(trn_idx)} val={len(val_idx)} R={R} dev={DEV}")

    model = Net(256).to(DEV)
    nump = sum(p.numel() for p in model.parameters())/1e6
    print(f"params {nump:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(trn_idx)//args.bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()
    post = torch.from_numpy(pos).to(DEV).float()        # (N,32)

    def get_batch(ii):
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)  # (b,32,1,192,192)
        if R != 192:
            b, t = x.shape[:2]
            x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(R, R), mode='bilinear', align_corners=False).reshape(b, t, 1, R, R)
        return x

    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); rs.shuffle(trn_idx); tl = 0; nb = 0
        for s in range(0, len(trn_idx), args.bs):
            ii = trn_idx[s:s+args.bs]
            x = get_batch(ii)
            x = augment(x, True)
            tgt = post[ii]                                # (b,32) abs position 0..47
            with torch.cuda.amp.autocast():
                sc, _ = model(x)
                reg = F.smooth_l1_loss(sc, tgt/47.0)      # normalized position regress
                rk = pairwise_rank_loss(sc, tgt)
                loss = reg + 0.5*rk
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
        if ep % 5 == 4 or ep == args.epochs-1:
            # quick val with default beta/scale
            S, rsq, msq, mf1 = score_split_resized(model, img, pos, rank, mask, val_idx, lp_present, R, beta=2.0, scale=1.0)
            print(f"ep{ep+1} loss{tl/nb:.3f} | val S={S:.4f} rank2={rsq:.4f} mask2={msq:.4f} mf1={mf1:.4f} | {time.time()-t0:.0f}s")
    # tune beta/scale on val
    print("tuning decode...")
    best = (-1, None)
    for beta in [0.5, 1, 2, 3, 5, 8]:
        for scale in [0.85, 1.0, 1.15]:
            S, rsq, msq, mf1 = score_split_resized(model, img, pos, rank, mask, val_idx, lp_present, R, beta=beta, scale=scale)
            if msq > best[0]: best = (msq, (beta, scale, S, rsq, msq, mf1))
    print("best decode:", best[1])
    torch.save({'model': model.state_dict(), 'lp_present': lp_present, 'R': R}, '_model_dev.pt')

def score_split_resized(model, img, pos, rank, mask, idx, lp_present, R, beta, scale, tta=True, bs=8):
    model.eval(); taus, f1s, f1sq = [], [], []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            ii = idx[s:s+bs]
            x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
            if R != 192:
                b, t = x.shape[:2]
                x = F.interpolate(x.reshape(b*t,1,192,192), size=(R,R), mode='bilinear', align_corners=False).reshape(b,t,1,R,R)
            sc, _ = model(x)
            if tta:
                sc2, _ = model(torch.flip(x, dims=[4])); sc = (sc+sc2)/2
            sc = sc.cpu().numpy()
            for bi, vi in enumerate(ii):
                s_k = sc[bi]
                pr = np.argsort(np.argsort(s_k))
                tau = kendalltau(pr, rank[vi]).statistic
                taus.append(max(0, tau)**2 if tau==tau else 0)
                order = np.argsort(s_k); q = s_k[order]
                q = (q-q.min())/(q.max()-q.min()+1e-6)*47
                m = decode_mask(q, lp_present, beta, scale)
                f = macro_f1_mask(m, mask[vi]); f1s.append(f); f1sq.append(f**2)
    rsq=float(np.mean(taus)); msq=float(np.mean(f1sq))
    return 0.6*rsq+0.4*msq, rsq, msq, float(np.mean(f1s))

if __name__ == '__main__':
    main()
