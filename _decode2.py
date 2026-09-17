import numpy as np, torch, torch.nn.functional as F, argparse
from scipy.stats import kendalltau
from sklearn.metrics import f1_score
from _train2 import Net, load_cache, build_targets, logsm, GMAX, DEV

def macro_f1(p, t): return f1_score(t, p, average='macro', labels=[0, 1], zero_division=0)
INF = 1e15

@torch.no_grad()
def dump(model, img, idx, R, flip_tta=True):
    model.eval(); POS=[]; GL=[]; LE=[]; TR=[]
    for s in range(0, len(idx), 8):
        ii = idx[s:s+8]
        x = torch.from_numpy(img[ii]).to(DEV).float().div(255).unsqueeze(2)
        if R != 192:
            b, t = x.shape[:2]
            x = F.interpolate(x.reshape(b*t,1,192,192), size=(R,R), mode='bilinear', align_corners=False).reshape(b,t,1,R,R)
        h = model.embed(x, flip_tta=flip_tta)
        pos = model.pos(h).cpu().numpy(); order = np.argsort(pos, axis=1)
        gl, le, tr = model.gaps(h, torch.from_numpy(order).to(DEV))
        POS.append(pos); GL.append(gl.cpu().numpy()); LE.append(le.cpu().numpy()); TR.append(tr.cpu().numpy())
    return np.concatenate(POS), np.concatenate(GL), np.concatenate(LE), np.concatenate(TR)

def gap_logp(gl_vol):  # (31,GMAX)->(31,GMAX) normalized logP
    return np.stack([logsm(gl_vol[k]) for k in range(31)])

def pos_dp(phat, gli, leadlp, traillp, lp_int_prior, lp_lead_prior, lp_trail_prior, lp_present,
           w_pos, w_gap, w_iprior, w_occ, w_lead, w_lprior):
    # assign 32 ordered slices to strictly-increasing slots 0..47; minimize cost
    # phat: (32,) calibrated abs pos; gli: (31,GMAX); leadlp/traillp: (GMAX,)
    A = 48
    gmaxcost = -np.log(1e-6)
    def gcvec(v):  # gap cost for adjacency before slice v (v>=1), g index = a-a'-1
        gc = np.full(A, gmaxcost*5.0)
        for g in range(A):
            gh = gli[v-1][g] if g < GMAX else gli[v-1][GMAX-1]-(g-GMAX+1)*2.0
            pr = lp_int_prior[g] if g < GMAX else lp_int_prior[GMAX-1]-(g-GMAX+1)*2.0
            gc[g] = -w_gap*gh - w_iprior*pr
        return gc
    node = -w_occ*lp_present  # (48,) prefer occupying present-likely slots
    # v=0
    dp = node.copy()
    for a in range(A):
        lh = leadlp[a] if a < GMAX else leadlp[GMAX-1]-(a-GMAX+1)*2.0
        lp = lp_lead_prior[a] if a < GMAX else lp_lead_prior[GMAX-1]-(a-GMAX+1)*2.0
        dp[a] += w_pos*(phat[0]-a)**2 - w_lead*lh - w_lprior*lp
    back = np.zeros((32, A), int)
    for v in range(1, 32):
        gc = gcvec(v)
        M = np.full((A, A), INF)
        for d in range(1, A):
            M[np.arange(A-d), np.arange(d, A)] = dp[:A-d] + gc[d-1]
        prev = np.argmin(M, axis=0); val = M[prev, np.arange(A)]
        nd = val + w_pos*(phat[v]-np.arange(A))**2 + node
        if v == 31:
            for a in range(A):
                t = 47-a
                th = traillp[t] if t < GMAX else traillp[GMAX-1]-(t-GMAX+1)*2.0
                tp = lp_trail_prior[t] if t < GMAX else lp_trail_prior[GMAX-1]-(t-GMAX+1)*2.0
                nd[a] += -w_lead*th - w_lprior*tp
        dp = nd; back[v] = prev
    a = int(np.argmin(dp)); slots = []
    for v in range(31, -1, -1):
        slots.append(a); a = back[v, a]
    slots = slots[::-1]
    m = np.ones(48, int); m[slots] = 0
    return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='_m2_seed0.pt'); ap.add_argument('--R', type=int, default=128); ap.add_argument('--nval', type=int, default=96)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    model = Net().to(DEV); model.load_state_dict(ck['model']); prior = ck['prior']
    img, vols = load_cache('train'); pos_t, rank, mask = build_targets('train.csv', vols)
    N=len(vols); perm=np.random.RandomState(0).permutation(N); val_idx=perm[:args.nval]; cal_idx=perm[args.nval:args.nval+150]

    POS,GL,LE,TR = dump(model, img, val_idx, args.R)
    PC,_,_,_ = dump(model, img, cal_idx, args.R)
    # calibrate phat: linear fit true_abs_pos ~ pred (per-slice, predicted order doesn't matter)
    xc = PC.reshape(-1); yc = pos_t[cal_idx].reshape(-1)
    A1 = np.polyfit(xc, yc, 1); print("calib slope/int", A1)
    def phat_of(p): return A1[0]*p + A1[1]
    lp_present = np.log((1-mask[cal_idx].mean(0)).clip(.02,.98)/mask[cal_idx].mean(0).clip(.02,.98))
    li, ll, lt = prior['internal'], prior['lead'], prior['trail']

    taus=[max(0,kendalltau(np.argsort(np.argsort(POS[i])),rank[val_idx[i]]).statistic)**2 for i in range(len(POS))]
    rank2=np.mean(taus); print(f"rank2={rank2:.4f}")
    M_=mask[val_idx]

    best=(-1,None)
    grid=[]
    for w_pos in [0.0, 0.05, 0.15, 0.4, 1.0]:
        for w_gap in [0.0, 1.0, 2.0]:
            for w_occ in [0.0, 0.5, 1.5]:
                for w_lead in [0.0, 1.0]:
                    for w_lprior in [0.0, 1.0, 2.0]:
                        w_iprior=0.3
                        f1s=[];ks=[]
                        for i in range(len(POS)):
                            gli=gap_logp(GL[i]); lelp=logsm(LE[i]); trlp=logsm(TR[i])
                            m=pos_dp(phat_of(POS[i]),gli,lelp,trlp,li,ll,lt,lp_present,
                                     w_pos,w_gap,w_iprior,w_occ,w_lead,w_lprior)
                            f1s.append(macro_f1(m,M_[i])); ks.append(int(((m==1)&(M_[i]==1)).sum()))
                        msq=np.mean(np.array(f1s)**2); mf=np.mean(f1s); mk=np.mean(ks)
                        if msq>best[0]: best=(msq,dict(w_pos=w_pos,w_gap=w_gap,w_occ=w_occ,w_lead=w_lead,w_lprior=w_lprior,mf1=mf,msq=msq,k=mk))
    print("BEST pos-DP:", best[1])
    print(f"=> est S = 0.6*{rank2:.4f} + 0.4*{best[0]:.4f} = {0.6*rank2+0.4*best[0]:.4f}")

if __name__=='__main__':
    main()
