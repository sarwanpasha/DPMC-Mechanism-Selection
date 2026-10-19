"""Gradient-perturbation baselines: DP-SGD (Gaussian, (eps, delta)) and DP-GD (Laplace, pure eps).

The learning rate and the number of epochs are selected on a validation split of the
true ratings for every seed. That is a non-private advantage given to the baseline on
purpose, which makes the comparison conservative for DPMC.
"""
import argparse

import numpy as np

import _common
from dpmc.data import load_cached, three_way_split, rmse, save
from dpmc.dpsgd import (dpsgd_mf, rdp_gaussian_eps, sigma_for_target_eps,
                        laplace_scale_pure)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--eps_list", type=str, default=_common.EPS)
    ap.add_argument("--delta", type=float, default=1e-6)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--reg", type=float, default=0.05)
    ap.add_argument("--epochs_list", type=str, default="25,50,100,200")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--lrs", type=str, default="1e-6,1e-5,1e-4,1e-3,1e-2")
    ap.add_argument("--noise", choices=["gauss", "laplace_pure"], default="gauss")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log
    cache_dir = args.cache_dir or args.data_dir
    rows, cols, vals, shape, r_min, r_max = load_cached(
        args.dataset, args.data_dir, args.m, args.n, args.data_seed, cache_dir)
    clip = (r_min, r_max)
    n_obs = int(vals.shape[0])
    eps_list = [float(x) for x in args.eps_list.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    ep_list = [int(x) for x in args.epochs_list.split(",")]
    method = "DPSGD-MF" if args.noise == "gauss" else "DPGD-MF-pure"
    payload = dict(meta=dict(dataset=args.dataset, shape=list(shape), n_obs_total=n_obs,
                             eps_list=eps_list, delta=args.delta, rank=args.rank, reg=args.reg,
                             epochs_list=ep_list, clipC=args.clip, lrs=lrs, seeds=args.seeds,
                             noise=args.noise, test_frac=args.test_frac, val_frac=args.val_frac),
                   records=[], runs=[])
    for eps in eps_list:
        def noise_for(T):
            if args.noise == "gauss":
                sg = sigma_for_target_eps(float(eps), T, args.delta)
                return sg, rdp_gaussian_eps(sg, T, args.delta), None
            return 0.0, float(eps), laplace_scale_pure(args.clip, args.rank, T, eps)

        for seed in range(args.seeds):
            tr, va, te = three_way_split(n_obs, args.val_frac, args.test_frac, seed)
            best = None
            for T in ep_list:
                sg, ct, bl = noise_for(T)
                for lr in lrs:
                    rng = np.random.default_rng([seed, int(round(eps * 1000)),
                                                 int(round(-np.log10(lr))), T])
                    pred = dpsgd_mf(rows[tr], cols[tr], vals[tr], shape, args.rank, args.reg,
                                    lr, T, args.clip, sg, rng, clip,
                                    noise_kind=args.noise, b_lap=bl)
                    v = float(rmse(pred(rows[va], cols[va]), vals[va]))
                    if best is None or v < best[0]:
                        best = (v, lr, pred, T, sg, ct)
            t_e = float(rmse(best[2](rows[te], cols[te]), vals[te]))
            payload["records"].append(dict(method=method, eps=float(eps), seed=int(seed),
                                           rmse=t_e, lr=float(best[1]), epochs=int(best[3]),
                                           sigma=float(best[4]), certified_eps=float(best[5]),
                                           val_rmse=float(best[0])))
            log("  [seed=%d eps=%-5s] lr=%g T=%d val=%.4f test=%.4f"
                % (seed, str(eps), best[1], best[3], best[0], t_e))
            save(args.out, payload)
    agg = {}
    for r in payload["records"]:
        agg.setdefault(r["eps"], []).append(r["rmse"])
    payload["runs"] = [dict(method=method, eps=k, rmse_mean=float(np.mean(v)),
                            rmse_std=float(np.std(v)), rmse_per_seed=v, n_seeds=len(v))
                       for k, v in agg.items()]
    save(args.out, payload)


if __name__ == "__main__":
    main()
