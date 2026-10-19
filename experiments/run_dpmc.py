"""DPMC on one mechanism across a list of budgets.

For every budget the mechanism privatizes the training ratings once per seed, the public
affine correction ``Xtilde = (Y - c) / alpha`` is applied, and the bias-aware
factorization is fitted by alternating least squares. ``(rank, lam)`` are chosen by
K-fold cross-validation on the debiased privatized values of the seed-0 release,
independently at each budget, so every budget is a separate eps-DP release followed by
post-processing. Test RMSE is measured on held-out true ratings.

Running with ``--mech rule`` selects the mechanism at each budget by the data-free rule
over the candidate list; any other value holds that mechanism fixed.

Output JSON: ``meta``, ``channel`` (per-budget statistics), ``select`` (cross-validation
tables), ``records`` (one per budget and seed) and ``runs`` (per-budget mean and std).
"""
import argparse
import time

import numpy as np

import _common
from dpmc import channels as CH
from dpmc.data import load_cached, grid_step, train_test_split, rmse, save
from dpmc.selection import release_rng, score_table, pick, fit_eval


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    _common.add_fit_args(ap)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eps_list", type=str, default=_common.EPS)
    ap.add_argument("--mech", choices=list(CH.ALL_MECHANISMS) + ["rule"], default="rule")
    ap.add_argument("--debias", type=int, default=1,
                    help="1: fit the corrected release; 0: fit the raw release")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log
    cache_dir = args.cache_dir or args.data_dir

    rows, cols, vals, shape, r_min, r_max = load_cached(
        args.dataset, args.data_dir, args.m, args.n, args.data_seed, cache_dir)
    clip = (r_min, r_max)
    n_obs = int(vals.shape[0])
    grid = CH.grid_for(r_min, r_max, grid_step(args.dataset))
    eps_list = sorted(float(x) for x in args.eps_list.split(","))
    ranks = [int(x) for x in args.ranks.split(",")]
    lams = sorted(float(x) for x in args.lams.split(","))
    log("[data] %s shape=%s n_obs=%d grid=%d levels mech=%s debias=%d folds=%d"
        % (args.dataset, str(shape), n_obs, grid.shape[0], args.mech, args.debias, args.folds))

    payload = dict(meta=dict(dataset=args.dataset, shape=list(shape), n_obs_total=n_obs,
                             r_min=r_min, r_max=r_max, grid=[float(g) for g in grid],
                             eps_list=eps_list, mech=args.mech, ranks=ranks, lams=lams,
                             n_iter=args.n_iter, seeds=args.seeds, debias=args.debias,
                             folds=args.folds, test_frac=args.test_frac,
                             selection="per-budget K-fold CV on privatized values"),
                   channel=[], select=[], records=[], runs=[])

    tr0, _ = train_test_split(n_obs, args.test_frac, 0)
    r0, c0, v0 = rows[tr0], cols[tr0], vals[tr0]
    plan = []
    for eps in eps_list:
        mech = (CH.select_mechanism(CH.CANDIDATES, grid, eps, r_min, r_max)
                if args.mech == "rule" else args.mech)
        alpha, V, s2, cc = CH.channel(mech, grid, eps, r_min, r_max)
        payload["channel"].append(dict(eps=float(eps), mech=mech, alpha=float(alpha),
                                       V=float(V), sigma2_eff=float(s2), intercept=float(cc)))
        log("[eps=%s] mech=%s alpha=%.5f V=%.5f sigma2_eff=%.4f" % (str(eps), mech, alpha, V, s2))
        y = CH.sample(mech, v0, grid, eps, r_min, r_max, release_rng(0, eps, mech))
        x = (y - cc) / alpha if args.debias else y
        S = score_table(r0, c0, x, shape, ranks, lams, args.n_iter, clip, 0, args.folds, log)
        chosen = pick(S, ranks, lams)
        payload["select"].append(dict(eps=float(eps), mech=mech, ranks=ranks, lams=lams,
                                      cv=[[float(z) for z in r] for r in S],
                                      rank=int(chosen[0]), lam=float(chosen[1])))
        log("[eps=%s] selected rank=%d lam=%g" % (str(eps), chosen[0], chosen[1]))
        plan.append((eps, mech, alpha, V, s2, cc, chosen))
        save(args.out, payload)

    for eps, mech, alpha, V, s2, cc, chosen in plan:
        for seed in range(args.seeds):
            tr, te = train_test_split(n_obs, args.test_frac, seed)
            r_tr, c_tr, v_tr = rows[tr], cols[tr], vals[tr]
            r_te, c_te, v_te = rows[te], cols[te], vals[te]
            y = CH.sample(mech, v_tr, grid, eps, r_min, r_max, release_rng(seed, eps, mech))
            x = (y - cc) / alpha if args.debias else y
            t0 = time.time()
            mdl = fit_eval(r_tr, c_tr, x, shape, chosen[0], chosen[1], args.n_iter, clip)
            e = float(rmse(mdl["predict"](r_te, c_te), v_te))
            payload["records"].append(dict(method="%s+mf" % mech, mech=mech, eps=float(eps),
                                           seed=int(seed), rmse=e, rank=int(chosen[0]),
                                           lam=float(chosen[1]), alpha=float(alpha),
                                           sigma2_eff=float(s2),
                                           secs=round(time.time() - t0, 1)))
            log("   [seed=%d eps=%-5s] rmse=%.4f (%.0fs)" % (seed, str(eps), e, time.time() - t0))
            save(args.out, payload)

    agg = {}
    for r in payload["records"]:
        agg.setdefault(r["eps"], []).append(r["rmse"])
    payload["runs"] = [dict(method=args.mech, eps=k, rmse_mean=float(np.mean(v)),
                            rmse_std=float(np.std(v)), rmse_per_seed=v, n_seeds=len(v))
                       for k, v in agg.items()]
    save(args.out, payload)
    log("=== SUMMARY (%s, %s) ===" % (args.dataset, args.mech))
    for r in sorted(payload["runs"], key=lambda z: z["eps"]):
        log("eps=%-5s %.4f +/- %.4f (%d)" % (str(r["eps"]), r["rmse_mean"], r["rmse_std"], r["n_seeds"]))


if __name__ == "__main__":
    main()
