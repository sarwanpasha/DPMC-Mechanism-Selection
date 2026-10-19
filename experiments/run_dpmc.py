"""DPMC on one mechanism (or the rule) for given budgets and seeds.

For every budget and seed the mechanism privatizes that seed's training ratings, the
public affine correction ``Xtilde = (Y - c) / alpha`` is applied, ``(rank, lam)`` are
chosen by K-fold cross-validation on that seed's debiased privatized training values, and
the bias-aware factorization is refitted on all of them. Every budget is therefore a
separate eps-DP release followed by post-processing, and no step reads a true rating
before the test evaluation. Test RMSE is measured on the held-out true ratings, together
with Recall@10 and NDCG@10 of ranking all items outside the user's training ratings
(relevant: held-out ratings of at least 4).

``--mech rule`` selects the mechanism at each budget by the data-free rule over the
candidate list; any other value holds that mechanism fixed. The hybrid below its
switching budget is the one-bit randomizer and is run as such (``channels.canonical``).

Output JSON: ``meta``, ``channel`` (per-budget statistics), ``select`` (one
cross-validation table per budget and seed) and ``records`` (one per budget and seed).
"""
import argparse
import time

import numpy as np

import _common
from dpmc import channels as CH
from dpmc.data import load_cached, grid_step, train_test_split, rmse, ranking_metrics, save
from dpmc.selection import release_rng, score_table, pick, fit_eval


def seeds_from(spec):
    """``"0-9"`` or ``"0,3,5"`` to a list of integers."""
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    _common.add_fit_args(ap)
    ap.add_argument("--seeds", type=str, default="0-9")
    ap.add_argument("--eps_list", type=str, default=_common.EPS)
    ap.add_argument("--mech", choices=list(CH.ALL_MECHANISMS) + ["rule"], default="rule")
    ap.add_argument("--debias", type=int, default=1,
                    help="1: fit the corrected release; 0: fit the raw release")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--relevant", type=float, default=4.0)
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
    seeds = seeds_from(args.seeds)
    ranks = [int(x) for x in args.ranks.split(",")]
    lams = sorted(float(x) for x in args.lams.split(","))
    log("[data] %s shape=%s n_obs=%d grid=%d levels mech=%s debias=%d folds=%d seeds=%s"
        % (args.dataset, str(shape), n_obs, grid.shape[0], args.mech, args.debias,
           args.folds, seeds))

    payload = dict(meta=dict(dataset=args.dataset, shape=list(shape), n_obs_total=n_obs,
                             r_min=r_min, r_max=r_max, grid=[float(g) for g in grid],
                             eps_list=eps_list, mech=args.mech, ranks=ranks, lams=lams,
                             n_iter=args.n_iter, seeds=seeds, debias=args.debias,
                             folds=args.folds, test_frac=args.test_frac, topk=args.topk,
                             relevant=args.relevant,
                             selection="per-seed, per-budget K-fold CV on privatized values"),
                   channel=[], select=[], records=[])

    for eps in eps_list:
        mech = (CH.select_mechanism(CH.CANDIDATES, grid, eps, r_min, r_max)
                if args.mech == "rule" else args.mech)
        mech = CH.canonical(mech, eps)
        alpha, V, s2, cc = CH.channel(mech, grid, eps, r_min, r_max)
        payload["channel"].append(dict(eps=float(eps), mech=mech, alpha=float(alpha),
                                       V=float(V), sigma2_eff=float(s2), intercept=float(cc)))
        log("[eps=%s] mech=%s alpha=%.5f V=%.5f sigma2_eff=%.4f" % (str(eps), mech, alpha, V, s2))
        for seed in seeds:
            tr, te = train_test_split(n_obs, args.test_frac, seed)
            r_tr, c_tr, v_tr = rows[tr], cols[tr], vals[tr]
            r_te, c_te, v_te = rows[te], cols[te], vals[te]
            y = CH.sample(mech, v_tr, grid, eps, r_min, r_max, release_rng(seed, eps, mech))
            x = (y - cc) / alpha if args.debias else y
            t0 = time.time()
            S = score_table(r_tr, c_tr, x, shape, ranks, lams, args.n_iter, clip, seed,
                            args.folds, log)
            rank, lam = pick(S, ranks, lams)
            mdl = fit_eval(r_tr, c_tr, x, shape, rank, lam, args.n_iter, clip)
            e = float(rmse(mdl["predict"](r_te, c_te), v_te))
            score = (mdl["mu"] + mdl["bu"][:, None] + mdl["bi"][None, :]
                     + mdl["P"] @ mdl["Q"].T)
            rec, ndcg, n_users = ranking_metrics(score, r_tr, c_tr, r_te, c_te, v_te,
                                                 args.topk, args.relevant)
            payload["select"].append(dict(eps=float(eps), seed=int(seed), mech=mech,
                                          ranks=ranks, lams=lams,
                                          cv=[[float(z) for z in r] for r in S],
                                          rank=int(rank), lam=float(lam)))
            payload["records"].append(dict(method="%s+mf" % mech, mech=mech, eps=float(eps),
                                           seed=int(seed), rmse=e, recall=rec, ndcg=ndcg,
                                           rank_users=n_users, rank=int(rank),
                                           lam=float(lam), alpha=float(alpha),
                                           sigma2_eff=float(s2),
                                           secs=round(time.time() - t0, 1)))
            log("   [seed=%d eps=%-5s] rank=%d lam=%g rmse=%.4f recall@%d=%.4f ndcg@%d=%.4f (%.0fs)"
                % (seed, str(eps), rank, lam, e, args.topk, rec, args.topk, ndcg,
                   time.time() - t0))
            save(args.out, payload)
    save(args.out, payload)


if __name__ == "__main__":
    main()
