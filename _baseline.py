import pandas as pd, numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import f1_score

df = pd.read_csv('train.csv', dtype={'volume_id': str})
vols = df['volume_id'].unique()
rng = np.random.RandomState(0)
perm = rng.permutation(len(vols))
val = set(vols[perm[:96]])          # ~20% val
trn = [v for v in vols if v not in val]

marr = np.array([[int(c) for c in s[1:]] for s in df.drop_duplicates('volume_id').set_index('volume_id').loc[trn]['true_missing_mask']])
marg = marr.mean(0)
top16 = np.zeros(48, int); top16[np.argsort(-marg)[:16]] = 1
print("train marginal top16 positions:", sorted(np.where(top16==1)[0].tolist()))

def macro_f1_mask(pred48, true48):
    return f1_score(true48, pred48, average='macro', labels=[0,1], zero_division=0)

# closed form check: (16+3k)/64 when both have 16 ones
def overlap_k(pred48, true48):
    return int(((pred48==1)&(true48==1)).sum())

# Eval constant-prior mask on val
ks=[]; f1s=[]
val_df = df[df['volume_id'].isin(val)].drop_duplicates('volume_id')
for _, r in val_df.iterrows():
    t = np.array([int(c) for c in r['true_missing_mask'][1:]])
    ks.append(overlap_k(top16, t))
    f1s.append(macro_f1_mask(top16, t))
ks=np.array(ks); f1s=np.array(f1s)
print(f"\nConstant top16 prior on val: mean overlap k={ks.mean():.2f}/16, mean macroF1={f1s.mean():.4f}, mean F1^2={np.mean(f1s**2):.4f}")
print(f"  -> closed form (16+3k)/64 with k={ks.mean():.2f}: {(16+3*ks.mean())/64:.4f}")

# threshold>0.5 (only 12 edge positions -> 12 ones)
thr = (marg>0.5).astype(int)
print("thr>0.5 positions:", sorted(np.where(thr==1)[0].tolist()), "count", thr.sum())
f1s2=[macro_f1_mask(thr, np.array([int(c) for c in r['true_missing_mask'][1:]])) for _,r in val_df.iterrows()]
print(f"thr>0.5 mask macroF1={np.mean(f1s2):.4f}, F1^2={np.mean(np.array(f1s2)**2):.4f}")

# ORACLE-ish: if we knew the true ORDER of visible slices and used ONLY edge prior for ends + filled
# internal gaps by marginal — skip; instead show perfect mask contributes 0.4, random rank contributes 0.
# Random rank score:
taus=[]
for v in list(val)[:40]:
    sub = df[df['volume_id']==v]
    tau = kendalltau(sub['presented_index'], sub['true_rank']).statistic  # presented order vs true = random
    taus.append(max(0,tau)**2)
print(f"\nRandom-order rank_score (presented_index as rank): mean={np.mean(taus):.4f}")
print("=> sample_submission est S ~= 0.6*~0 + 0.4*%.4f = %.4f" % (np.mean(f1s**2), 0.4*np.mean(f1s**2)))
print("=> If we get rank tau=0.90 (score .81) and mask prior-only: S ~= 0.6*0.81 + 0.4*%.4f = %.4f"%(np.mean(f1s**2),0.6*0.81+0.4*np.mean(f1s**2)))
print("=> Need strong mask: e.g. mask F1=0.80 (F1^2=.64): S=0.6*0.81+0.4*0.64=%.4f"%(0.6*0.81+0.4*0.64))
