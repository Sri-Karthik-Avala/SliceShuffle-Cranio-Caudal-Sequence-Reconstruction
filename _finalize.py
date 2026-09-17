import numpy as np, pandas as pd
from scipy.stats import kendalltau
from solution import unified_dp, logsm1, logsm2, macro_f1, find_root, GMAX

d = np.load('_logits.npz', allow_pickle=True)
pri = {'lead': d['pri_lead'], 'trail': d['pri_trail'], 'internal': d['pri_internal']}
lp_present = d['lp_present']
hP, hC, hG, hL, hT = d['hP'], d['hC'], d['hG'], d['hL'], d['hT']
hrank, hmask = d['hold_rank'], d['hold_mask']
tP, tC, tG, tL, tT = d['tP'], d['tC'], d['tG'], d['tL'], d['tT']
te_vols = [int(v) for v in d['te_vols']]

taus = [max(0, kendalltau(np.argsort(np.argsort(hP[i])), hrank[i]).statistic)**2 for i in range(len(hP))]
rank2 = float(np.mean(taus)); print(f"holdout rank2={rank2:.4f}  (n={len(hP)})")

def dp_masks(P, C, G, L, T, W):
    out = []
    for i in range(len(P)):
        o = np.argsort(P[i])
        out.append(unified_dp(logsm2(C[i])[o], logsm2(G[i]), logsm1(L[i]), logsm1(T[i]), pri, **W, lp_present=lp_present))
    return out

def soft_occ_masks(C):
    out = []
    for i in range(len(C)):
        occ = np.exp(logsm2(C[i])).sum(0); miss = np.argsort(occ)[:16]; m = np.zeros(48, int); m[miss] = 1; out.append(m)
    return out

grid = [dict(w_cls=wc, w_gap=wg, w_iprior=wi, w_lead=wl, w_lprior=wp, w_occ=wo)
        for wc in [0.5, 1.0, 2.0, 3.0, 4.0] for wg in [0.5, 1.0, 2.0] for wi in [0.2, 0.4]
        for wl in [0.0, 1.0] for wp in [0.5, 1.5] for wo in [0.0, 0.5]]
best = (-1, None)
for W in grid:
    ms = dp_masks(hP, hC, hG, hL, hT, W)
    msq = np.mean([macro_f1(ms[i], hmask[i])**2 for i in range(len(ms))])
    if msq > best[0]: best = (msq, W)
mf = np.mean([macro_f1(m, hmask[i]) for i, m in enumerate(dp_masks(hP, hC, hG, hL, hT, best[1]))])
print(f"UNIFIED-DP best mask2={best[0]:.4f} mf1={mf:.4f} S={0.6*rank2+0.4*best[0]:.4f} W={best[1]}")

soc = soft_occ_masks(hC); soc_msq = np.mean([macro_f1(soc[i], hmask[i])**2 for i in range(len(soc))])
print(f"SOFT-OCC mask2={soc_msq:.4f}")

use_W = best[1]; method = 'dp'
if soc_msq > best[0] + 0.003:
    method = 'soft'; print("=> choosing SOFT-OCC")
else:
    print("=> choosing UNIFIED-DP")

# apply to test
if method == 'dp':
    te_masks = dp_masks(tP, tC, tG, tL, tT, use_W)
else:
    te_masks = soft_occ_masks(tC)
pred_rank = {te_vols[i]: np.argsort(np.argsort(tP[i])) for i in range(len(te_vols))}
mask_str = {te_vols[i]: 'M'+''.join(str(int(b)) for b in te_masks[i]) for i in range(len(te_vols))}
for i in range(len(te_vols)): assert te_masks[i].sum() == 16

root = find_root(); te_df = pd.read_csv(root+'test.csv'); te_df['volume_id'] = te_df['volume_id'].astype(int)
rows = []
for _, r in te_df.iterrows():
    v = int(r['volume_id']); p = int(r['presented_index'])
    rows.append((int(r['row_id']), v, p, int(pred_rank[v][p]), mask_str[v]))
out = pd.DataFrame(rows, columns=['row_id', 'volume_id', 'presented_index', 'pred_rank', 'pred_missing_mask']).sort_values('row_id').reset_index(drop=True)
import os; os.makedirs('working', exist_ok=True); out.to_csv('working/submission.csv', index=False)
print(f"REWROTE working/submission.csv ({method}) rows={len(out)} est_S={0.6*rank2+0.4*max(best[0],soc_msq if method=='soft' else best[0]):.4f}")
