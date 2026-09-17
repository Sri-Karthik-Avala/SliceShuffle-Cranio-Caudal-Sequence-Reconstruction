"""Train ONE model at resolution R / seed, save its holdout+test logits to _ens_R{R}_s{seed}.npz.
Reuses all model/training code from solution.py. Lets us accumulate seeds/resolutions for ensembling
without retraining, and run one job at a time (thermally gentle)."""
import os, sys, time, argparse, numpy as np, pandas as pd, torch
import solution as S

def load_imgs(root, split, vols):
    cache = f'_cache_{split}_img.npy'
    if os.path.exists(cache):
        a = np.load(cache)
        if a.shape[0] == len(vols):
            return a
    return S.load_split(root, split, vols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, default=192); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=64); ap.add_argument('--hold', type=int, default=80)
    a = ap.parse_args()
    t0 = time.time(); root = S.find_root()
    tr_df = pd.read_csv(root+'train.csv'); tr_df['volume_id'] = tr_df['volume_id'].astype(int)
    te_df = pd.read_csv(root+'test.csv'); te_df['volume_id'] = te_df['volume_id'].astype(int)
    tr_vols = sorted(tr_df['volume_id'].unique()); te_vols = sorted(te_df['volume_id'].unique())
    img = load_imgs(root, 'train', tr_vols); te_img = load_imgs(root, 'test', te_vols)
    pos, rank, mask = S.build_targets(tr_df, tr_vols)
    N = len(tr_vols); perm = np.random.RandomState(0).permutation(N); hold = perm[:a.hold]; trn = perm[a.hold:]
    pri, int_w, lp_present = S.build_priors(mask[trn]); cls_w = torch.tensor(int_w, dtype=torch.float32, device=S.DEV); soft = S.make_soft_targets(pos)
    print(f"R={a.R} seed={a.seed} epochs={a.epochs} train={len(trn)} hold={len(hold)} dev={S.DEV}", flush=True)
    m = S.train_one(a.seed, img, pos, rank, trn, cls_w, soft, a.R, a.epochs)
    print(f"trained {time.time()-t0:.0f}s", flush=True)
    torch.save({'model': m.state_dict(), 'R': a.R, 'seed': a.seed}, f'_w_R{a.R}_s{a.seed}.pt')
    hP, hC, hG, hL, hT = S.predict(m, img, hold, a.R)
    tP, tC, tG, tL, tT = S.predict(m, te_img, np.arange(len(te_vols)), a.R)
    np.savez_compressed(f'_ens_R{a.R}_s{a.seed}.npz', hP=hP, hC=hC, hG=hG, hL=hL, hT=hT,
                        tP=tP, tC=tC, tG=tG, tL=tL, tT=tT)
    print(f"saved _w_R{a.R}_s{a.seed}.pt + _ens_R{a.R}_s{a.seed}.npz  total {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
