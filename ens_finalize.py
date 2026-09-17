"""Ensemble all _ens_*.npz logits, tune decode W (with an honest cross-holdout estimate to avoid
W-overfit), write working/submission.csv. Reuses solution.py decode."""
import os, glob, argparse, numpy as np, pandas as pd
from scipy.stats import kendalltau
import solution as S

def macro_f1(p, t): return S.macro_f1(p, t)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--hold', type=int, default=80)
    ap.add_argument('--files', default='_ens_*.npz'); ap.add_argument('--write', type=int, default=1)
    a = ap.parse_args()
    files = sorted(glob.glob(a.files)); assert files, "no _ens_*.npz files"
    print("ensembling:", files)
    K = ['hP', 'hC', 'hG', 'hL', 'hT', 'tP', 'tC', 'tG', 'tL', 'tT']
    acc = {k: None for k in K}
    for f in files:
        d = np.load(f)
        for k in K: acc[k] = d[k].astype(np.float64) if acc[k] is None else acc[k]+d[k]
    for k in K: acc[k] /= len(files)
    hP, hC, hG, hL, hT = acc['hP'], acc['hC'], acc['hG'], acc['hL'], acc['hT']
    tP, tC, tG, tL, tT = acc['tP'], acc['tC'], acc['tG'], acc['tL'], acc['tT']

    root = S.find_root()
    tr_df = pd.read_csv(root+'train.csv'); tr_df['volume_id'] = tr_df['volume_id'].astype(int)
    te_df = pd.read_csv(root+'test.csv'); te_df['volume_id'] = te_df['volume_id'].astype(int)
    tr_vols = sorted(tr_df['volume_id'].unique()); te_vols = sorted(te_df['volume_id'].unique())
    pos, rank, mask = S.build_targets(tr_df, tr_vols)
    N = len(tr_vols); perm = np.random.RandomState(0).permutation(N); hold = perm[:a.hold]
    hrank, hmask = rank[hold], mask[hold]
    pri, _, lp = S.build_priors(mask)   # full-data priors

    def masks_on(P, C, G, L, T, idxs, W):
        out = []
        for i in idxs:
            o = np.argsort(P[i]); out.append(S.unified_dp(S.logsm2(C[i])[o], S.logsm2(G[i]), S.logsm1(L[i]), S.logsm1(T[i]), pri, **W, lp_present=lp))
        return out
    def mask2_on(idxs, W):
        ms = masks_on(hP, hC, hG, hL, hT, idxs, W)
        return np.mean([macro_f1(ms[j], hmask[idxs[j]])**2 for j in range(len(idxs))])
    def bestW(idxs):
        b = (-1, S.WGRID[0])
        for W in S.WGRID:
            v = mask2_on(idxs, W)
            if v > b[0]: b = (v, W)
        return b[1]

    taus = [max(0, kendalltau(np.argsort(np.argsort(hP[i])), hrank[i]).statistic)**2 for i in range(len(hP))]
    rank2 = float(np.mean(taus))

    alli = np.arange(len(hold))
    half = len(alli)//2; A, B = alli[:half], alli[half:]
    WA = bestW(A); estB = mask2_on(B, WA)
    WB = bestW(B); estA = mask2_on(A, WB)
    honest_mask2 = (estA+estB)/2
    W_full = bestW(alli); full_mask2 = mask2_on(alli, W_full)
    # split-robust W: maximize the worse of the two halves (generalizes better than argmax-full)
    W_rob = max(S.WGRID, key=lambda W: min(mask2_on(A, W), mask2_on(B, W))); rob_mask2 = mask2_on(alli, W_rob)
    W_use = W_rob
    print(f"rank2={rank2:.4f}")
    print(f"mask2: full(optimistic)={full_mask2:.4f}  robust-W full={rob_mask2:.4f}  honest(cross-holdout)={honest_mask2:.4f}")
    print(f"S: full={0.6*rank2+0.4*full_mask2:.4f}  HONEST={0.6*rank2+0.4*honest_mask2:.4f}")
    print(f"W_full={W_full}\nW_use(robust)={W_use}")

    if a.write:
        tmasks = masks_on(tP, tC, tG, tL, tT, list(range(len(te_vols))), W_use)
        for m in tmasks: assert m.sum() == 16
        pr = {te_vols[i]: np.argsort(np.argsort(tP[i])) for i in range(len(te_vols))}
        ms = {te_vols[i]: 'M'+''.join(str(int(b)) for b in tmasks[i]) for i in range(len(te_vols))}
        rows = []
        for _, r in te_df.iterrows():
            v = int(r['volume_id']); p = int(r['presented_index'])
            rows.append((int(r['row_id']), v, p, int(pr[v][p]), ms[v]))
        out = pd.DataFrame(rows, columns=['row_id', 'volume_id', 'presented_index', 'pred_rank', 'pred_missing_mask']).sort_values('row_id').reset_index(drop=True)
        os.makedirs('working', exist_ok=True); out.to_csv('working/submission.csv', index=False)
        print(f"WROTE working/submission.csv rows={len(out)} from {len(files)} models")

if __name__ == '__main__':
    main()
