"""Search model subsets for the best honest (cross-holdout) S, then write the winning submission."""
import glob, numpy as np, pandas as pd
from scipy.stats import kendalltau
import solution as S

def load(files):
    K = ['hP','hC','hG','hL','hT','tP','tC','tG','tL','tT']; acc={k:None for k in K}
    for f in files:
        d=np.load(f)
        for k in K: acc[k]=d[k].astype(np.float64) if acc[k] is None else acc[k]+d[k]
    for k in K: acc[k]/=len(files)
    return acc

root=S.find_root()
tr_df=pd.read_csv(root+'train.csv'); tr_df['volume_id']=tr_df['volume_id'].astype(int)
te_df=pd.read_csv(root+'test.csv'); te_df['volume_id']=te_df['volume_id'].astype(int)
tr_vols=sorted(tr_df['volume_id'].unique()); te_vols=sorted(te_df['volume_id'].unique())
pos,rank,mask=S.build_targets(tr_df,tr_vols)
N=len(tr_vols); perm=np.random.RandomState(0).permutation(N); hold=perm[:80]
hrank,hmask=rank[hold],mask[hold]; pri,_,lp=S.build_priors(mask)

def masks_on(P,C,G,L,T,idxs,W):
    return [S.unified_dp(S.logsm2(C[i])[np.argsort(P[i])],S.logsm2(G[i]),S.logsm1(L[i]),S.logsm1(T[i]),pri,**W,lp_present=lp) for i in idxs]
def mask2(acc,idxs,W):
    ms=masks_on(acc['hP'],acc['hC'],acc['hG'],acc['hL'],acc['hT'],idxs,W)
    return float(np.mean([S.macro_f1(ms[j],hmask[idxs[j]])**2 for j in range(len(idxs))]))
def evalsub(acc):
    n=len(acc['hP']); A=list(range(40)); B=list(range(40,80))
    WA=max(S.WGRID,key=lambda w:mask2(acc,A,w)); estB=mask2(acc,B,WA)
    WB=max(S.WGRID,key=lambda w:mask2(acc,B,w)); estA=mask2(acc,A,WB)
    honest=(estA+estB)/2
    Wrob=max(S.WGRID,key=lambda w:min(mask2(acc,A,w),mask2(acc,B,w)))
    full=mask2(acc,list(range(n)),Wrob)
    taus=[max(0,kendalltau(np.argsort(np.argsort(acc['hP'][i])),hrank[i]).statistic)**2 for i in range(n)]
    r2=float(np.mean(taus))
    return r2,full,honest,Wrob

R160=sorted(glob.glob('_ens_R160_*.npz')); R192=sorted(glob.glob('_ens_R192_*.npz'))
cands={
 'R160x4':R160, 'R192x5':R192,
 '3+3':R160[:3]+R192[:3], '3+2':R160[:3]+R192[:2], '4+3':R160[:4]+R192[:3],
 '4+4':R160[:4]+R192[:4], '4+2':R160[:4]+R192[:2], '3+4':R160[:3]+R192[:4],
 'all9':R160+R192,
}
best=(-1,None,None)
for name,files in cands.items():
    if len(files)<2: continue
    acc=load(files); r2,full,honest,W=evalsub(acc)
    Sh=0.6*r2+0.4*honest; Sf=0.6*r2+0.4*full
    print(f"{name:7s} n={len(files)} rank2={r2:.4f} mask2_full={full:.4f} mask2_honest={honest:.4f}  S_full={Sf:.4f} S_honest={Sh:.4f}")
    if Sh>best[0]: best=(Sh,name,(files,W))
print(f"\nBEST: {best[1]}  S_honest={best[0]:.4f}")

# write submission for best subset
files,W=best[2]; acc=load(files)
tm=[S.unified_dp(S.logsm2(acc['tC'][i])[np.argsort(acc['tP'][i])],S.logsm2(acc['tG'][i]),S.logsm1(acc['tL'][i]),S.logsm1(acc['tT'][i]),pri,**W,lp_present=lp) for i in range(len(te_vols))]
for m in tm: assert m.sum()==16
pr={te_vols[i]:np.argsort(np.argsort(acc['tP'][i])) for i in range(len(te_vols))}
ms={te_vols[i]:'M'+''.join(str(int(b)) for b in tm[i]) for i in range(len(te_vols))}
rows=[(int(r['row_id']),int(r['volume_id']),int(r['presented_index']),int(pr[int(r['volume_id'])][int(r['presented_index'])]),ms[int(r['volume_id'])]) for _,r in te_df.iterrows()]
out=pd.DataFrame(rows,columns=['row_id','volume_id','presented_index','pred_rank','pred_missing_mask']).sort_values('row_id').reset_index(drop=True)
import os; os.makedirs('working',exist_ok=True); out.to_csv('working/submission.csv',index=False)
print(f"WROTE working/submission.csv from BEST subset '{best[1]}' ({len(files)} models) W={W}")
