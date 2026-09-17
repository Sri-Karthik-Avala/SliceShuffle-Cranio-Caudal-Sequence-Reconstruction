import pandas as pd, numpy as np, sys
sub = pd.read_csv('working/submission.csv')
te = pd.read_csv('test.csv')
errs = []
# columns
exp = ['row_id', 'volume_id', 'presented_index', 'pred_rank', 'pred_missing_mask']
if list(sub.columns) != exp: errs.append(f"columns {list(sub.columns)} != {exp}")
# row_id set
if set(sub['row_id']) != set(te['row_id']): errs.append("row_id set mismatch with test.csv")
if len(sub) != len(te): errs.append(f"row count {len(sub)} != {len(te)}")
# duplicate (volume_id, presented_index)
if sub.duplicated(['volume_id', 'presented_index']).any(): errs.append("duplicate (volume_id,presented_index)")
# pred_rank range + permutation per volume
pr = sub['pred_rank']
if pr.min() < 0 or pr.max() > 31: errs.append(f"pred_rank out of [0,31]: {pr.min()}..{pr.max()}")
badperm = 0
for v, g in sub.groupby('volume_id'):
    if sorted(g['pred_rank'].tolist()) != list(range(32)): badperm += 1
if badperm: errs.append(f"{badperm} volumes have non-permutation pred_rank")
# mask checks
badmask = 0; badones = 0; badident = 0
for v, g in sub.groupby('volume_id'):
    masks = g['pred_missing_mask'].astype(str).unique()
    if len(masks) != 1: badident += 1
    m = masks[0]
    if len(m) != 49 or m[0] != 'M' or any(c not in '01' for c in m[1:]): badmask += 1
    elif m[1:].count('1') != 16: badones += 1
if badident: errs.append(f"{badident} volumes have non-identical masks across rows")
if badmask: errs.append(f"{badmask} volumes have malformed mask (need 'M'+48 binary)")
if badones: errs.append(f"{badones} volumes have !=16 ones")

print("rows:", len(sub), "volumes:", sub['volume_id'].nunique())
print("sample mask ones:", sub.groupby('volume_id')['pred_missing_mask'].first().head(3).apply(lambda s: str(s)[1:].count('1')).tolist())
if errs:
    print("FAIL:"); [print("  -", e) for e in errs]; sys.exit(1)
print("ALL CHECKS PASSED")
