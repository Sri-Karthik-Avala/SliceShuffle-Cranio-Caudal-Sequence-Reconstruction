import numpy as np, pandas as pd, os, glob
from PIL import Image
import time

def build(split):
    df = pd.read_csv(f'{split}.csv', dtype={'volume_id': str})
    vols = sorted(df['volume_id'].unique(), key=lambda x: int(x))
    N = len(vols)
    arr = np.zeros((N, 32, 192, 192), np.uint8)
    t0 = time.time()
    for vi, v in enumerate(vols):
        for s in range(32):
            p = f'{split}/{int(v):04d}/s{s:02d}.png'
            arr[vi, s] = np.asarray(Image.open(p).convert('L'), np.uint8)
    np.save(f'_cache_{split}_img.npy', arr)
    np.save(f'_cache_{split}_vols.npy', np.array(vols))
    print(split, "imgs", arr.shape, arr.nbytes/1e6, "MB", round(time.time()-t0,1), "s")
    return vols

for sp in ['train', 'test']:
    build(sp)
print("done")
