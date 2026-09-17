import numpy as np, torch, argparse
from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from _train3 import Net, load_cache, build_targets, predict, unified_dp, logsm1, logsm2, GMAX, DEV

def macro_f1(p, t): return f1_score(t, p, average='macro', labels=[0, 1], zero_division=0)

def soft_occ_mask(cls_lp):  # cls_lp (32,48) log-prob (any order)
    occ = np.exp(cls_lp).sum(0)             # expected occupancy per position
    miss = np.argsort(occ)[:16]             # 16 least-occupied -> missing
    m = np.zeros(48, int); m[miss] = 1; return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='_m3_seed0.pt'); ap.add_argument('--R', type=int, default=160); ap.add_argument('--nval', type=int, default=96)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=DEV, weights_only=False)
    model = Net().to(DEV); model.load_state_dict(ck['model']); pri = ck['prior']; lp_present = ck['lp_present']
    img, vols = load_cache('train'); pos_t, rank, mask = build_targets('train.csv', vols)
    N = len(vols); perm = np.random.RandomState(0).permutation(N); val_idx = perm[:a.nval]
    POS, CLS, GL, LE, TR = predict(model, img, val_idx, a.R)
    R_, M_ = rank[val_idx], mask[val_idx]
    taus = [max(0, kendalltau(np.argsort(np.argsort(POS[i])), R_[i]).statistic)**2 for i in range(len(POS))]
    rank2 = np.mean(taus); print(f"rank2={rank2:.4f}")

    # unified DP grid (incl w_cls=0 = gap-only)
    grid = [dict(w_cls=wc, w_gap=wg, w_iprior=0.3, w_lead=wl, w_lprior=wp, w_occ=wo)
            for wc in [0.0, 0.5, 1.0, 2.0, 4.0] for wg in [0.0, 1.0, 2.0] for wl in [0.0, 1.0] for wp in [0.5, 1.5] for wo in [0.0, 0.5]]
    best = (-1, None, 0)
    for W in grid:
        f1s = []; ks = []
        for i in range(len(POS)):
            o = np.argsort(POS[i]); m = unified_dp(logsm2(CLS[i])[o], logsm2(GL[i]), logsm1(LE[i]), logsm1(TR[i]), pri, **W, lp_present=lp_present)
            f1s.append(macro_f1(m, M_[i])); ks.append(int(((m == 1) & (M_[i] == 1)).sum()))
        msq = np.mean(np.array(f1s)**2)
        if msq > best[0]: best = (msq, W, np.mean(f1s), np.mean(ks))
    print(f"UNIFIED-DP best mask2={best[0]:.4f} mf1={best[2]:.4f} k={best[3]:.2f} W={best[1]}")
    print(f"  => S={0.6*rank2+0.4*best[0]:.4f}")

    # soft-occupancy
    f1s = [macro_f1(soft_occ_mask(logsm2(CLS[i])), M_[i]) for i in range(len(POS))]
    print(f"SOFT-OCC mask2={np.mean(np.array(f1s)**2):.4f} mf1={np.mean(f1s):.4f}")

    # prior-only
    marg = mask[perm[a.nval:]].mean(0); pm = np.zeros(48, int); pm[np.argsort(-marg)[:16]] = 1
    f1p = [macro_f1(pm, M_[i]) for i in range(len(POS))]
    print(f"PRIOR-ONLY mask2={np.mean(np.array(f1p)**2):.4f} mf1={np.mean(f1p):.4f}")

if __name__ == '__main__':
    main()
