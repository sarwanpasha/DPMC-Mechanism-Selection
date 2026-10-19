"""Non-private reference ceiling and simple baselines on the same blocks and splits.

Phase A selects ``(rank, reg)`` of the biased factorization on a validation split of seed
0; phase B refits the winner for every seed and scores it on the test split, together with
the global-mean predictor and the additive bias-only model.
"""
import argparse
import time

import numpy as np

import _common
from dpmc.data import load_cached, three_way_split, rmse, save
from dpmc.mf import fit_biased_mf, fit_bias_only


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--ranks", type=str, default="8,16,32")
    ap.add_argument("--regs", type=str, default="3,10,30")
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log
    cache_dir = args.cache_dir or args.data_dir
    rows, cols, vals, shape, r_min, r_max = load_cached(
        args.dataset, args.data_dir, args.m, args.n, args.data_seed, cache_dir)
    clip = (r_min, r_max)
    ranks = [int(x) for x in args.ranks.split(",")]
    regs = [float(x) for x in args.regs.split(",")]
    n_obs = int(vals.shape[0])
    payload = dict(meta=dict(dataset=args.dataset, shape=list(shape), n_obs_total=n_obs,
                             r_min=r_min, r_max=r_max, ranks=ranks, regs=regs,
                             n_iter=args.n_iter, val_frac=args.val_frac,
                             test_frac=args.test_frac, seeds=args.seeds),
                   grid=[], records=[], runs=[])
    tr, va, _ = three_way_split(n_obs, args.val_frac, args.test_frac, 0)
    best = None
    for rk in ranks:
        for rg in regs:
            t0 = time.time()
            mdl = fit_biased_mf(rows[tr], cols[tr], vals[tr], shape, rank=rk, reg=rg,
                                n_iter=args.n_iter, clip=clip)
            v_e = float(rmse(mdl["predict"](rows[va], cols[va]), vals[va]))
            log("[A seed=0] rank=%3d reg=%6.1f val=%.4f (%.1fs)" % (rk, rg, v_e, time.time() - t0))
            payload["grid"].append(dict(rank=rk, reg=rg, val_rmse=v_e))
            if best is None or v_e < best[0]:
                best = (v_e, rk, rg)
    v_best, rk_best, rg_best = best
    payload["meta"]["selected"] = dict(rank=rk_best, reg=rg_best, val_rmse=v_best)
    for seed in range(args.seeds):
        tr, va, te = three_way_split(n_obs, args.val_frac, args.test_frac, seed)
        R, C, V = rows[tr], cols[tr], vals[tr]
        tr_, tc, tv = rows[te], cols[te], vals[te]
        mu = float(V.mean())
        payload["records"].append(dict(method="GlobalMean", seed=seed,
                                       rmse=float(rmse(np.full(tv.shape, mu), tv))))
        bo = fit_bias_only(R, C, V, shape, clip=clip)
        payload["records"].append(dict(method="BiasOnly", seed=seed,
                                       rmse=float(rmse(bo["predict"](tr_, tc), tv))))
        mdl = fit_biased_mf(R, C, V, shape, rank=rk_best, reg=rg_best, n_iter=args.n_iter, clip=clip)
        t_e = float(rmse(mdl["predict"](tr_, tc), tv))
        payload["records"].append(dict(method="NonPrivate-BiasedMF", seed=seed, rmse=t_e,
                                       rank=rk_best, reg=rg_best))
        log("[B seed=%d] test=%.4f" % (seed, t_e))
    agg = {}
    for rec in payload["records"]:
        agg.setdefault(rec["method"], []).append(rec["rmse"])
    payload["runs"] = [dict(method=k, rmse_mean=float(np.mean(v)), rmse_std=float(np.std(v)),
                            rmse_per_seed=v, n_seeds=len(v)) for k, v in agg.items()]
    save(args.out, payload)
    for r in payload["runs"]:
        log("%-22s %.4f +/- %.4f" % (r["method"], r["rmse_mean"], r["rmse_std"]))


if __name__ == "__main__":
    main()
