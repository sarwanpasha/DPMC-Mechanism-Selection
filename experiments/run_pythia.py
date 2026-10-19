"""Budget-spending, data-dependent mechanism selection in the style of Pythia.

Pythia (Kotsogiannis et al., SIGMOD 2017) spends part of the privacy budget on private
features of the data and selects an algorithm from them. Here the feature is the rating
histogram and the selector is the effective-noise criterion itself, evaluated under the
estimated law instead of the uniform one, which is the strongest selector of that form:

1. spend ``f * eps`` on a Laplace histogram of the training ratings. Under the
   rating-level relation with the observation pattern public, changing one rating moves
   one unit between two bins, so the l1 sensitivity is 2 and the noise scale is
   ``2 / (f * eps)``;
2. clip the noisy counts at zero and normalize them into ``pi_hat``;
3. select ``argmin_Q sigma2_eff(Q; pi_hat)`` over the same candidate list, at the budget
   that remains, ``(1 - f) * eps``;
4. privatize with the selected mechanism at ``(1 - f) * eps`` and reconstruct exactly as
   DPMC does.

The two releases compose sequentially to ``eps``; everything after them is
post-processing. The data-free rule spends nothing on step 1 and releases at the full
budget. Hyperparameters are cross-validated per seed on that seed's release, as in
``run_dpmc.py``, and the same error and ranking metrics are reported.
"""
import argparse
import time

import numpy as np

import _common
from dpmc import channels as CH
from dpmc.data import load_cached, grid_step, train_test_split, rmse, ranking_metrics, save
from dpmc.selection import stable_hash, score_table, pick, fit_eval


def private_histogram(v, grid, eps_s, rng):
    """Laplace histogram of ratings on the grid, clipped at zero and normalized."""
    k = len(grid)
    idx = np.abs(v[:, None] - grid[None, :]).argmin(axis=1)
    counts = np.bincount(idx, minlength=k).astype(float)
    noisy = np.maximum(counts + rng.laplace(0.0, 2.0 / eps_s, size=k), 0.0)
    if noisy.sum() <= 0.0:
        return np.full(k, 1.0 / k)
    return noisy / noisy.sum()


def selector_rng(seed, eps, frac):
    """Generator for the histogram noise of ``(seed, eps, frac)``."""
    return np.random.default_rng([seed, int(round(eps * 1000)), int(round(frac * 1000)), 4242])


def release_rng_pythia(seed, eps, frac, mech):
    """Generator for the privatization draw of the selected mechanism."""
    return np.random.default_rng([seed, int(round(eps * 1000)), int(round(frac * 1000)),
                                  abs(stable_hash(mech)) % 1000, 17])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    _common.add_fit_args(ap)
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--eps_list", type=str, default=_common.EPS)
    ap.add_argument("--fracs", type=str, default="0.05,0.1,0.2",
                    help="fractions of the budget spent on the histogram")
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
    fracs = [float(x) for x in args.fracs.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    lams = sorted(float(x) for x in args.lams.split(","))
    seeds = []
    for part in args.seeds.split(","):
        a, _, b = part.partition("-")
        seeds.extend(range(int(a), int(b or a) + 1))
    payload = dict(meta=dict(dataset=args.dataset, shape=list(shape), n_obs_total=n_obs,
                             eps_list=eps_list, fracs=fracs, ranks=ranks, lams=lams,
                             n_iter=args.n_iter, folds=args.folds, seeds=seeds,
                             candidates=list(CH.CANDIDATES), test_frac=args.test_frac),
                   select=[], records=[])
    for eps in eps_list:
        rule = CH.select_mechanism(CH.CANDIDATES, grid, eps, r_min, r_max)
        for frac in fracs:
            eps_s, eps_r = frac * eps, (1.0 - frac) * eps
            for seed in seeds:
                tr, te = train_test_split(n_obs, args.test_frac, seed)
                r_tr, c_tr, v_tr = rows[tr], cols[tr], vals[tr]
                r_te, c_te, v_te = rows[te], cols[te], vals[te]
                pi_hat = private_histogram(v_tr, grid, eps_s, selector_rng(seed, eps, frac))
                mech = CH.select_mechanism(CH.CANDIDATES, grid, eps_r, r_min, r_max, pi=pi_hat)
                alpha, V, s2, cc = CH.channel(mech, grid, eps_r, r_min, r_max)
                y = CH.sample(mech, v_tr, grid, eps_r, r_min, r_max,
                              release_rng_pythia(seed, eps, frac, mech))
                x = (y - cc) / alpha
                t0 = time.time()
                S = score_table(r_tr, c_tr, x, shape, ranks, lams, args.n_iter, clip, seed,
                                args.folds, log)
                rank, lam = pick(S, ranks, lams)
                payload["select"].append(dict(eps=eps, frac=frac, seed=seed, mech=mech,
                                              rank=int(rank), lam=float(lam),
                                              cv=[[float(z) for z in r] for r in S]))
                mdl = fit_eval(r_tr, c_tr, x, shape, rank, lam, args.n_iter, clip)
                e = float(rmse(mdl["predict"](r_te, c_te), v_te))
                score = (mdl["mu"] + mdl["bu"][:, None] + mdl["bi"][None, :]
                         + mdl["P"] @ mdl["Q"].T)
                rec, ndcg, _ = ranking_metrics(score, r_tr, c_tr, r_te, c_te, v_te)
                payload["records"].append(dict(eps=eps, frac=frac, seed=seed, mech=mech,
                                               rule=rule, eps_select=eps_s, eps_release=eps_r,
                                               pi_hat=[float(p) for p in pi_hat], rmse=e,
                                               recall=rec, ndcg=ndcg,
                                               rank=int(rank), lam=float(lam),
                                               secs=round(time.time() - t0, 1)))
                log("[eps=%-5g f=%-4g seed=%d] selected=%-5s rule=%-5s rmse=%.4f ndcg=%.4f"
                    % (eps, frac, seed, mech, rule, e, ndcg))
                save(args.out, payload)
    save(args.out, payload)
    log("[done] %s" % args.out)


if __name__ == "__main__":
    main()
