"""Nuclear-norm reconstruction of the same corrected release, the reconstruction ablation.

With ``--bias 1`` the additive part ``mu + b_i + c_j`` is fitted first by ridge regression
on the corrected release and the residual is completed by nuclear-norm minimization; with
``--bias 0`` the corrected release is completed directly. The weight is either fixed
(``--lam_fixed``) or chosen on a held-out slice of the privatized release
(``--lam_mults``, multipliers of the noise-calibrated default).
"""
import argparse
import time

import numpy as np

import _common
from dpmc import channels as CH
from dpmc.data import load_cached, grid_step, train_test_split, rmse, save
from dpmc.mf import fit_bias_only
from dpmc.nuclear import admm_nuclear, default_lambda


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps_list", type=str, default=_common.EPS)
    ap.add_argument("--mech", choices=list(CH.ALL_MECHANISMS) + ["rule"], default="krr")
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--fidelity", type=str, default="l1")
    ap.add_argument("--lam_fixed", type=float, default=0.0)
    ap.add_argument("--lam_mults", default="")
    ap.add_argument("--sel_iters", type=int, default=0)
    ap.add_argument("--admm_iters", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log
    cache_dir = args.cache_dir or args.data_dir
    rows, cols, vals, shape, r_min, r_max = load_cached(
        args.dataset, args.data_dir, args.m, args.n, args.data_seed, cache_dir)
    delta = r_max - r_min
    n_obs = int(vals.shape[0]); m, n = shape
    grid = CH.grid_for(r_min, r_max, grid_step(args.dataset))
    eps_list = [float(x) for x in args.eps_list.split(",")]
    payload = dict(meta=dict(dataset=args.dataset, shape=list(shape), n_obs_total=n_obs,
                             eps_list=eps_list, mech=args.mech, bias=args.bias,
                             fidelity=args.fidelity, lam_fixed=args.lam_fixed,
                             lam_mults=args.lam_mults, admm_iters=args.admm_iters,
                             seeds=args.seeds, test_frac=args.test_frac),
                   records=[], runs=[])
    for eps in eps_list:
        mech = (CH.select_mechanism(CH.CANDIDATES, grid, eps, r_min, r_max)
                if args.mech == "rule" else args.mech)
        alpha, V, s2, cc = CH.channel(mech, grid, eps, r_min, r_max)
        for seed in range(args.seeds):
            tr, te = train_test_split(n_obs, args.test_frac, seed)
            r_tr, c_tr, v_tr = rows[tr], cols[tr], vals[tr]
            r_te, c_te, v_te = rows[te], cols[te], vals[te]
            rng = np.random.default_rng([seed, int(round(eps * 1000)), 77])
            y = CH.sample(mech, v_tr, grid, eps, r_min, r_max, rng)
            x = (y - cc) / alpha
            t0 = time.time()
            if args.bias:
                bo = fit_bias_only(r_tr, c_tr, x, shape, clip=None)
                base_te = bo["predict"](r_te, c_te)
                resid = x - bo["predict"](r_tr, c_tr)
            else:
                base_te = np.zeros(v_te.shape[0]); resid = x
            lam0 = default_lambda(delta, eps, resid.shape[0], m, n)
            sel_scores = {}
            if args.lam_fixed > 0:
                lam = args.lam_fixed
            elif args.lam_mults:
                cand = [lam0 * float(z) for z in args.lam_mults.split(",")]
                perm = rng.permutation(resid.shape[0])
                nva = max(1, int(0.1 * resid.shape[0]))
                va, tr2 = perm[:nva], perm[nva:]
                best, lam = float("inf"), lam0
                for lc in cand:
                    Xc = admm_nuclear(r_tr[tr2], c_tr[tr2], resid[tr2], shape, lc,
                                      fidelity=args.fidelity,
                                      max_iter=(args.sel_iters or args.admm_iters))[0]
                    sc = float(rmse(Xc[r_tr[va], c_tr[va]], resid[va]))
                    sel_scores[float(lc)] = sc
                    if sc < best:
                        best, lam = sc, lc
            else:
                lam = lam0
            X = admm_nuclear(r_tr, c_tr, resid, shape, lam, fidelity=args.fidelity,
                             max_iter=args.admm_iters)[0]
            pred = np.clip(base_te + X[r_te, c_te], r_min, r_max)
            e = float(rmse(pred, v_te))
            payload["records"].append(dict(method="nn%s" % ("+bias" if args.bias else ""),
                                           eps=float(eps), seed=int(seed), rmse=e, mech=mech,
                                           lam=float(lam), sel=sel_scores,
                                           secs=round(time.time() - t0, 1)))
            log("   [seed=%d eps=%-5s] rmse=%.4f" % (seed, str(eps), e))
            save(args.out, payload)
    agg = {}
    for r in payload["records"]:
        agg.setdefault(r["eps"], []).append(r["rmse"])
    payload["runs"] = [dict(method=payload["records"][0]["method"], eps=k,
                            rmse_mean=float(np.mean(v)), rmse_std=float(np.std(v)),
                            rmse_per_seed=v, n_seeds=len(v)) for k, v in agg.items()]
    save(args.out, payload)


if __name__ == "__main__":
    main()
