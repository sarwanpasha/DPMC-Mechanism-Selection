"""Controlled study of the selection frontier as a function of the number of levels ``k``.

Ground truth is a rank-``d_true`` matrix plus user and item effects, quantized onto a
``k``-level grid on ``[1, 5]``. Everything downstream is the DPMC pipeline, so the only
thing that varies across runs is ``k``.
"""
import argparse
import time

import numpy as np

import _common
from dpmc import channels as CH
from dpmc.mf import fit_biased_mf
from dpmc.data import save
from dpmc.selection import stable_hash

LO, HI = 1.0, 5.0


def make_data(m, n, k, d_true, density, seed):
    """Synthetic observations quantized onto ``k`` equally spaced levels."""
    rng = np.random.default_rng(1000 + seed)
    P = rng.normal(0.0, 1.0, (m, d_true))
    Q = rng.normal(0.0, 1.0, (n, d_true))
    bu = rng.normal(0.0, 0.45, m)
    bi = rng.normal(0.0, 0.45, n)
    S = 3.0 + bu[:, None] + bi[None, :] + (P @ Q.T) / np.sqrt(d_true)
    grid = np.linspace(LO, HI, k)
    nnz = int(round(density * m * n))
    idx = rng.choice(m * n, size=nnz, replace=False)
    r, c = np.unravel_index(idx, (m, n))
    v = np.clip(S[r, c], LO, HI)
    j = np.abs(v[:, None] - grid[None, :]).argmin(1)
    return r.astype(np.int64), c.astype(np.int64), grid[j], grid


def cv_select(r, c, v, shape, ranks, lams, n_iter, folds, seed):
    """``(rank, lam)`` minimizing K-fold RMSE on the corrected privatized values."""
    rng = np.random.default_rng(seed + 977)
    fold = rng.integers(0, folds, size=v.shape[0])
    best = None
    for d in ranks:
        for lam in lams:
            se = 0.0
            cnt = 0
            for f in range(folds):
                tr = fold != f
                te = ~tr
                M = fit_biased_mf(r[tr], c[tr], v[tr], shape, rank=d, reg=lam,
                                  n_iter=n_iter, clip=None)
                p = M["predict"](r[te], c[te])
                se += float(np.sum((p - v[te]) ** 2))
                cnt += int(te.sum())
            s = float(np.sqrt(se / cnt))
            if best is None or s < best[0]:
                best = (s, d, lam)
    return best[1], best[2]


def rmse(a, b):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--m", type=int, default=1200)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--d_true", type=int, default=8)
    ap.add_argument("--density", type=float, default=0.20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test_frac", type=float, default=0.10)
    ap.add_argument("--eps_list", type=str, default=_common.EPS)
    ap.add_argument("--mechs", type=str, default="krr,geom,laplace,shin,expo,pm,hm")
    ap.add_argument("--ranks", type=str, default="8,16")
    ap.add_argument("--lams", type=str, default="1,10,100,1000,10000,100000")
    ap.add_argument("--n_iter", type=int, default=12)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log
    eps_list = [float(x) for x in args.eps_list.split(",")]
    mechs = args.mechs.split(",")
    ranks = [int(x) for x in args.ranks.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    shape = (args.m, args.n)
    r, c, v, grid = make_data(args.m, args.n, args.k, args.d_true, args.density, 0)
    rs = np.random.default_rng(7)
    te_mask = rs.random(v.shape[0]) < args.test_frac
    r_tr, c_tr, v_tr = r[~te_mask], c[~te_mask], v[~te_mask]
    r_te, c_te, v_te = r[te_mask], c[te_mask], v[te_mask]
    payload = dict(meta=vars(args), grid=grid.tolist(), n_train=int(v_tr.shape[0]),
                   n_test=int(v_te.shape[0]), channel=[], records=[])
    ref = fit_biased_mf(r_tr, c_tr, v_tr, shape, rank=16, reg=10.0, n_iter=args.n_iter, clip=(LO, HI))
    payload["nonprivate"] = rmse(ref["predict"](r_te, c_te), v_te)
    for mech in mechs:
        for eps in eps_list:
            t0 = time.time()
            alpha, V, s2, cc = CH.channel(mech, grid, eps, LO, HI)
            payload["channel"].append(dict(mech=mech, eps=eps, alpha=alpha, V=V, sigma2_eff=s2, c=cc))
            rng0 = np.random.default_rng([abs(stable_hash(mech)) % 9973, int(eps * 1000)])
            x0 = (CH.sample(mech, v_tr, grid, eps, LO, HI, rng0) - cc) / alpha
            d, lam = cv_select(r_tr, c_tr, x0, shape, ranks, lams, args.n_iter, args.folds, 0)
            for seed in range(args.seeds):
                rng = np.random.default_rng([abs(stable_hash(mech)) % 9973, int(eps * 1000), seed])
                x = (CH.sample(mech, v_tr, grid, eps, LO, HI, rng) - cc) / alpha
                M = fit_biased_mf(r_tr, c_tr, x, shape, rank=d, reg=lam, n_iter=args.n_iter, clip=(LO, HI))
                payload["records"].append(dict(mech=mech, eps=eps, seed=seed,
                                               rmse=rmse(M["predict"](r_te, c_te), v_te),
                                               rank=d, lam=lam, sigma2_eff=s2, alpha=alpha))
            vv = [z["rmse"] for z in payload["records"] if z["mech"] == mech and z["eps"] == eps]
            log("[k=%d] %-8s eps=%-5s s2eff=%10.4g rmse=%.4f+-%.4f d=%d lam=%g (%.0fs)"
                % (args.k, mech, eps, s2, np.mean(vv), np.std(vv, ddof=1), d, lam, time.time() - t0))
            save(args.out, payload)
    save(args.out, payload)


if __name__ == "__main__":
    main()
