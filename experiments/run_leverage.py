"""Mean of t(x)^2 under the leverage law of the ALS half-steps.

The risk identity weights a rating by ``w_i = a_i^T G^2 a_i`` with
``G = (A^T A + lam I)^{-1}``, where ``A`` stacks the rows ``(q_j, 1)`` of the user's items
(user half-step) or ``(p_i, 1)`` of the item's users (item half-step). The one-bit and
piecewise mechanisms have per-level quantities affine in ``t(x)^2``, so the mean of
``t^2`` under these weights decides which of the two has the smaller effective noise under
the leverage law. The design is taken from a non-private fit to the seed-0 training split;
the weights depend on the design only.
"""
import argparse

import numpy as np

import _common
from dpmc.data import load_cached, train_test_split, save
from dpmc.mf import fit_biased_mf


def leverage(rows, cols, F, n, lam, size):
    """Leverage weight of every rating for the half-step that solves for ``rows``."""
    W = np.zeros(size)
    order = np.argsort(rows, kind="stable")
    for ids in np.split(order, np.cumsum(np.bincount(rows, minlength=n))[:-1]):
        if ids.size == 0:
            continue
        A = np.hstack([F[cols[ids]], np.ones((ids.size, 1))])
        G = np.linalg.inv(A.T @ A + lam * np.eye(A.shape[1]))
        W[ids] = np.einsum("ij,jk,ik->i", A, G @ G, A)
    return W


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lams", type=str, default="30,100,300,1000")
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache_dir = args.cache_dir or args.data_dir
    rows, cols, vals, shape, lo, hi = load_cached(
        args.dataset, args.data_dir, args.m, args.n, args.data_seed, cache_dir)
    tr, _ = train_test_split(int(vals.shape[0]), args.test_frac, 0)
    r, c, v = rows[tr], cols[tr], vals[tr]
    t2 = (2.0 * (v - lo) / (hi - lo) - 1.0) ** 2
    payload = dict(meta=dict(dataset=args.dataset, rank=args.rank, n_train=int(v.size),
                             uniform_grid_mean_t2=None, marginal_mean_t2=float(t2.mean())),
                   records=[])
    for lam in [float(x) for x in args.lams.split(",")]:
        M = fit_biased_mf(r, c, v, shape, rank=args.rank, reg=lam, n_iter=args.n_iter, clip=None)
        wu = leverage(r, c, M["Q"], shape[0], lam, v.size)
        wi = leverage(c, r, M["P"], shape[1], lam, v.size)
        rec = dict(lam=lam, user_half=float(wu @ t2 / wu.sum()), item_half=float(wi @ t2 / wi.sum()))
        payload["records"].append(rec)
        _common.log("[%s lam=%g] marginal %.4f user-half %.4f item-half %.4f"
                    % (args.dataset, lam, t2.mean(), rec["user_half"], rec["item_half"]))
    k = int(round((hi - lo) / (0.5 if args.dataset.startswith("ml-25m") else 1.0))) + 1
    payload["meta"]["uniform_grid_mean_t2"] = (k + 1) / (3.0 * (k - 1))
    save(args.out, payload)


if __name__ == "__main__":
    main()
