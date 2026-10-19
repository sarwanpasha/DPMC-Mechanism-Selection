"""How far the noise-correlated ALS design moves a half-step from the fixed-design identity.

The risk identity holds for a design independent of the privatization noise. In ALS the
design of a user's half-step is the item factor matrix, which was itself fitted to a
release that contains that user's ratings. Let ``(A, o)`` be the design and offset
``o_j = mu + c_j`` produced by ALS on the full release, and ``(A0, o0)`` those produced by
the same procedure, from the same initialization, on the release with the user's row
removed. ``(A0, o0)`` is independent of that user's privatization noise, so the
identity applies exactly to ``theta0 = (A0^T A0 + lam I)^{-1} A0^T (x - o0)``. For the
estimate the procedure actually computes, ``theta = (A^T A + lam I)^{-1} A^T (x - o)``,

    theta - theta0 = (A^T A + lam I)^{-1} (E^T r0 - A^T (E theta0 + delta)),

with ``E = A - A0``, ``delta = o - o0`` and ``r0 = x - o0 - A0 theta0``, hence

    ||theta - theta0|| <= (||E|| ||r0|| + ||A|| (||E|| ||theta0|| + ||delta||)) / (s_min(A)^2 + lam).

For a sample of users this script measures ``||theta - theta0||``, the bound, and the
privacy-noise standard deviation of ``theta0`` given by the identity,
``s = sqrt(sum_j v(x_j) w_j)`` with ``w_j = a0_j^T (A0^T A0 + lam I)^{-2} a0_j`` and
``v(x) = Var(Y | X = x) / alpha^2``. The reported ratio ``||theta - theta0|| / s`` says how
large the design correlation is relative to the error the identity describes.
Hyperparameters are the cross-validated ones of the corresponding ``run_dpmc.py`` run.
"""
import argparse

import numpy as np

import _common
from dpmc import channels as CH
from dpmc.data import load_cached, grid_step, train_test_split, save
from dpmc.mf import fit_biased_mf, user_half_step
from dpmc.results import hyperparameters
from dpmc.selection import release_rng, stable_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    _common.add_data_args(ap)
    ap.add_argument("--mech", choices=list(CH.CANDIDATES), required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--hparams", nargs="+", required=True,
                    help="result files or directories of run_dpmc.py for this dataset")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log
    cache_dir = args.cache_dir or args.data_dir

    rows, cols, vals, shape, r_min, r_max = load_cached(
        args.dataset, args.data_dir, args.m, args.n, args.data_seed, cache_dir)
    clip = (r_min, r_max)
    grid = CH.grid_for(r_min, r_max, grid_step(args.dataset))
    rank, lam = hyperparameters(args.hparams)[(args.dataset, args.mech, round(args.eps, 4))]
    alpha, V, s2, cc = CH.channel(args.mech, grid, args.eps, r_min, r_max)
    vlev = CH.conditional_variance(args.mech, grid, args.eps, r_min, r_max)

    tr, _ = train_test_split(int(vals.shape[0]), args.test_frac, 0)
    r_tr, c_tr, v_tr = rows[tr], cols[tr], vals[tr]
    y = CH.sample(args.mech, v_tr, grid, args.eps, r_min, r_max, release_rng(0, args.eps, args.mech))
    x = (y - cc) / alpha
    lev = np.abs(v_tr[:, None] - grid[None, :]).argmin(axis=1)

    full = fit_biased_mf(r_tr, c_tr, x, shape, rank=rank, reg=lam, n_iter=args.n_iter, clip=clip)
    counts = np.bincount(r_tr, minlength=shape[0])
    eligible = np.flatnonzero(counts >= rank + 2)
    pick_rng = np.random.default_rng([0, int(round(args.eps * 1000)),
                                      abs(stable_hash(args.mech)) % 1000, 77])
    users = np.sort(pick_rng.choice(eligible, size=min(args.users, eligible.size), replace=False))
    log("[loo] %s %s eps=%g rank=%d lam=%g alpha=%.4f sigma2_eff=%.4g users=%d"
        % (args.dataset, args.mech, args.eps, rank, lam, alpha, s2, users.size))

    out = dict(meta=dict(dataset=args.dataset, mech=args.mech, eps=args.eps, rank=rank,
                         lam=lam, alpha=alpha, sigma2_eff=s2, n_iter=args.n_iter,
                         users=int(users.size), shape=list(shape)),
               records=[])
    for u in users:
        own = r_tr == u
        loo = fit_biased_mf(r_tr[~own], c_tr[~own], x[~own], shape, rank=rank, reg=lam,
                            n_iter=args.n_iter, clip=clip)
        cols_u, x_u = c_tr[own], x[own]
        th, A, yt = user_half_step(full["Q"], full["bi"], full["mu"], cols_u, x_u, lam)
        th0, A0, y0 = user_half_step(loo["Q"], loo["bi"], loo["mu"], cols_u, x_u, lam)
        reg_I = lam * np.eye(A.shape[1])
        E = A - A0
        delta = (y0 - yt)
        r0 = y0 - A0 @ th0
        pred = np.linalg.solve(A.T @ A + reg_I, E.T @ r0 - A.T @ (E @ th0 + delta))
        gap = float(np.linalg.norm(th - th0))
        ident = float(np.linalg.norm(pred - (th - th0)))
        smin = float(np.linalg.svd(A, compute_uv=False).min())
        bound = (np.linalg.norm(E, 2) * np.linalg.norm(r0)
                 + np.linalg.norm(A, 2) * (np.linalg.norm(E, 2) * np.linalg.norm(th0)
                                           + np.linalg.norm(delta))) / (smin ** 2 + lam)
        Ginv = np.linalg.inv(A0.T @ A0 + reg_I)
        w = np.einsum("ij,jk,ik->i", A0, Ginv @ Ginv, A0)
        sd = float(np.sqrt(np.sum(vlev[lev[own]] * w)))
        out["records"].append(dict(user=int(u), n_u=int(own.sum()), gap=gap, bound=float(bound),
                                   identity_residual=ident, noise_sd=sd, ratio=gap / sd,
                                   theta0_norm=float(np.linalg.norm(th0))))
        log("   user=%5d n=%4d gap=%.3e bound=%.3e sd=%.3e ratio=%.4f ident=%.1e"
            % (u, own.sum(), gap, bound, sd, gap / sd, ident))
        save(args.out, out)
    ratios = np.array([r["ratio"] for r in out["records"]])
    out["summary"] = dict(median_ratio=float(np.median(ratios)),
                          p95_ratio=float(np.percentile(ratios, 95)),
                          max_ratio=float(ratios.max()),
                          bound_holds=bool(all(r["gap"] <= r["bound"] * (1 + 1e-9) + 1e-15
                                               for r in out["records"])),
                          max_identity_residual=float(max(r["identity_residual"]
                                                          for r in out["records"])))
    save(args.out, out)
    log("[summary] %s" % out["summary"])


if __name__ == "__main__":
    main()
