import pandas as pd, numpy as np, glob
from PIL import Image

df = pd.read_csv('train.csv', dtype={'volume_id': str})
print("cols", list(df.columns), "shape", df.shape)
print(df.dtypes.to_dict())

masks = df.drop_duplicates('volume_id')['true_missing_mask']
arr = np.array([[int(c) for c in s[1:]] for s in masks])
print("\nmask matrix", arr.shape, "ones/vol mean", arr.sum(1).mean(), "min", arr.sum(1).min(), "max", arr.sum(1).max())
marg = arr.mean(0)
print("\nMarginal P(missing=1) per position 0..47:")
for i in range(0, 48, 12):
    print(i, np.round(marg[i:i+12], 3))

# how often is each position present? present = visible slice there
print("\nMost-missing positions (top16 by marginal):", sorted(np.argsort(-marg)[:16].tolist()))

# Gap structure: for each vol, the visible positions (mask==0) in order; gaps between consecutive
def gap_stats(arr):
    edge_first=[]; edge_last=[]; internal_gaps=[]
    for row in arr:
        present = np.where(row==0)[0]
        edge_first.append(present[0])               # missing before first visible
        edge_last.append(47-present[-1])            # missing after last visible
        d = np.diff(present)-1                       # missing between consecutive visible
        internal_gaps.extend(d.tolist())
    return np.array(edge_first), np.array(edge_last), np.array(internal_gaps)
ef, el, ig = gap_stats(arr)
print("\nMissing before first visible: mean", ef.mean(), "max", ef.max(), "dist", np.bincount(ef, minlength=10)[:10])
print("Missing after last visible:  mean", el.mean(), "max", el.max(), "dist", np.bincount(el, minlength=10)[:10])
print("Internal gap sizes between consecutive visible (count by size):", np.bincount(ig, minlength=8)[:8])
print("Total internal missing mean per vol:", 16-ef.mean()-el.mean())

# rank labels per volume
g = df.groupby('volume_id')
v0 = df['volume_id'].iloc[0]
print("\nranks vol", v0, sorted(g.get_group(v0)['true_rank'].tolist())[:8], "... permutation of 0..31?",
      sorted(g.get_group(v0)['true_rank'].tolist()) == list(range(32)))

# images
fs = glob.glob('train/*/s00.png')
im = Image.open(fs[0]); a = np.array(im)
print("\nimg", im.mode, im.size, a.dtype, a.shape, "min", a.min(), "max", a.max(), "mean", round(a.mean(),1))
# check a few sizes
sizes=set()
for f in fs[:50]:
    sizes.add(Image.open(f).size)
print("unique sizes sample", sizes)
