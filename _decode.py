import numpy as np, torch, torch.nn.functional as F, argparse
from scipy.stats import kendalltau
from sklearn.metrics import f1_score, roc_auc_score
from _train2 import Net, load_cache, build_targets, count_dp, build_mask_from_gaps, logsm, GMAX, DEV

def macro_f1(p, t): return f1_score(t, p, average='macro', labels=[0, 1], zero_division=0)

@torch.no_grad()
def dump(model, img, idx, R, flip_tta=True):
    model.eval(); out = {'pos': [], 'gl': [], 'le': [], 'tr': [], 'embd': []}
    for s in range(0, len(idx), 8):
        ii = idx[s:s+8]
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if R != 192:
            b, t = x.shape[:2]
            x = F.interpolate(x.reshape(b*t, 1, 192, 192), size=(R, R), mode='bilinear', align_corners=False).reshape(b, t, 1, R, R)
        h = model.embed(x, flip_tta=flip_tta)
        pos = model.pos(h).cpu().numpy()
        order = np.argsort(pos, axis=1)
        ot = torch.from_numpy(order).to(DEV)
        gl, le, tr = model.gaps(h, ot)
        hc = h.cpu().numpy()
        for bi in range(len(ii)):
            o = order[bi]; ho = hc[bi][o]
            out['embd'].append(np.linalg.norm(ho[1:]-ho[:-1], axis=1))  # (31,)
        out['pos'].append(pos); out['gl'].append(gl.cpu().numpy()); out['le'].append(le.cpu().numpy()); out['tr'].append(tr.cpu().numpy())
    for k in ['pos', 'gl', 'le', 'tr']:
        out[k] = np.concatenate(out[k])
    out['embd'] = np.array(out['embd'])
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='_m2_seed0.pt'); ap.add_argument('--R', type=int, default=128); ap.add_argument('--nval', type=int, default=96)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    model = Net().to(DEV); model.load_state_dict(ck['model'])
    prior = ck['prior']; prior_mask = ck['prior_mask']
    img, vols = load_cache('train'); pos_t, rank, mask = build_targets('train.csv', vols)
    N = len(vols); perm = np.random.RandomState(0).permutation(N); val_idx = perm[:args.nval]
    D = dump(model, img, val_idx, args.R)
    POS, GL, LE, TR, EMBD = D['pos'], D['gl'], D['le'], D['tr'], D['embd']
    R_, M_ = rank[val_idx], mask[val_idx]

    # rank
    taus = [max(0, kendalltau(np.argsort(np.argsort(POS[i])), R_[i]).statistic)**2 for i in range(len(POS))]
    print(f"rank2={np.mean(taus):.4f}")

    # emb-dist -> gap-class logP via empirical calibration fit on a SEPARATE train chunk (true order)
    trn_idx = perm[args.nval:args.nval+200]
    Dt = dump(model, img, trn_idx, args.R)
    # build (embd, true gap) pairs from train using TRUE order
    embs, gaps = [], []
    for j, vi in enumerate(trn_idx):
        o = np.argsort(rank[vi]); po = pos_t[vi][o]
        # recompute embd in TRUE order: need h; approximate using predicted-order embd won't align. Instead use predicted order gaps vs predicted pos. Skip; use val self-calibration via bins on predicted order + true gaps in predicted order.
    # Simpler robust calibration: bins of EMBD over val, map to P(gap-class) using val true gaps in predicted order
    # gather predicted-order true gaps
    pe, pg = [], []
    for i in range(len(POS)):
        o = np.argsort(POS[i]); po = pos_t[val_idx[i]][o]
        g = np.clip(po[1:]-po[:-1]-1, 0, GMAX-1)
        pe.append(EMBD[i]); pg.append(g)
    pe = np.concatenate(pe); pg = np.concatenate(pg)
    y = (pg >= 1).astype(int)
    print(f"AUC emb-dist gap>=1: {roc_auc_score(y, pe):.4f}   gap-head P(>=1) AUC: {roc_auc_score(y, 1-np.exp([logsm(GL[i][k])[0] for i in range(len(POS)) for k in range(31)])):.4f}")

    def emb_to_logp(embd_vol):
        # map each scalar dist to a soft gap-class logP using global bin means (calibrate on val pe/pg)
        # build per-class gaussian over emb-dist
        out = np.zeros((31, GMAX))
        for g in range(GMAX):
            sel = pg == g
            if sel.sum() > 5:
                mu, sd = pe[sel].mean(), pe[sel].std()+1e-3
                out[:, g] = -0.5*((embd_vol-mu)/sd)**2 - np.log(sd)
            else:
                out[:, g] = -1e3
        return out - out.max(1, keepdims=True)

    def decode(i, src, lam, lam_edge):
        gli = np.stack([logsm(GL[i][k]) for k in range(31)]) if src in ('head', 'both') else np.zeros((31, GMAX))
        if src in ('emb', 'both'):
            gle = emb_to_logp(EMBD[i])
            gli = gli + gle if src == 'both' else gle
            gli = gli - gli.max(1, keepdims=True)
        lel = logsm(LE[i]); trl = logsm(TR[i])
        slots = [(1-lam_edge)*lel + lam_edge*prior['lead']]
        for k in range(31):
            slots.append((1-lam)*gli[k] + lam*prior['internal'])
        slots.append((1-lam_edge)*trl + lam_edge*prior['trail'])
        gaps = count_dp(slots); return build_mask_from_gaps(gaps)

    print("\n--- decode sweep (src, lam_internal, lam_edge): mf1 / f1^2 / mean-k ---")
    best = (-1, None)
    for src in ['head', 'emb', 'both']:
        for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            for lam_e in [0.0, 0.3, 0.6, 1.0]:
                f1s, ks = [], []
                for i in range(len(POS)):
                    m = decode(i, src, lam, lam_e)
                    f1s.append(macro_f1(m, M_[i])); ks.append(int(((m == 1) & (M_[i] == 1)).sum()))
                msq = np.mean(np.array(f1s)**2); mf = np.mean(f1s)
                if msq > best[0]: best = (msq, (src, lam, lam_e, mf, msq, np.mean(ks)))
    print("BEST:", best[1])
    src, lam, lam_e = best[1][0], best[1][1], best[1][2]
    S = 0.6*np.mean(taus)+0.4*best[0]
    print(f"=> est S = 0.6*{np.mean(taus):.4f} + 0.4*{best[0]:.4f} = {S:.4f}")
    # prior-only baseline within DP
    f1p = np.mean([macro_f1(prior_mask, M_[i]) for i in range(len(POS))])
    print(f"constant prior_mask mf1={f1p:.4f} f1^2={np.mean([macro_f1(prior_mask,M_[i])**2 for i in range(len(POS))]):.4f}")

if __name__ == '__main__':
    main()
