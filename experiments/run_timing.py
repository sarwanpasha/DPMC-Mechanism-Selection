"""Wall-clock cost of each stage of DPMC, and of the alternatives it is compared with.

Stages, each timed as the median of repeated runs on one block:

* rule selection   closed-form channel statistics of the five candidates and the argmin;
                   reads only the grid and the budget;
* Pythia-style     a Laplace histogram of the ratings plus selection under the estimated
  selection        law (see run_pythia.py);
* release          privatizing every training rating with the selected mechanism;
* correction       the public affine map ``(Y - c) / alpha``;
* reconstruction   one ALS fit of the biased factorization at the reported rank and ridge;
* LP bound / RR-on-bins  solving the optimal-channel programs of dpmc.optimal (once per
                   grid and budget; independent of the data).

Cross-validation multiplies the reconstruction cost by the number of fits it performs,
``folds * len(ranks) * len(lams)``, each on ``(folds - 1) / folds`` of the data; the
script reports that count alongside the single-fit time.
"""
import argparse
import platform
import time

import numpy as np

import _common
from dpmc import channels as CH, optimal, closed_forms as CF
from dpmc.data import load_cached, grid_step, train_test_split, save
from dpmc.mf import fit_biased_mf
from dpmc.selection import release_rng


def timeit(fn, reps):
    """Median wall-clock seconds of ``fn()`` over ``reps`` calls, and the last result."""
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--blocks", type=str, default="ml-25m:2000,ml-25m:4000,ml-25m:8000,amazon:2000",
                    help="comma list of dataset:size, the block being size x size")
    ap.add_argument("--ml_dir", required=True, help="directory with MovieLens caches or raw data")
    ap.add_argument("--am_dir", required=True, help="directory with Amazon caches or raw data")
    ap.add_argument("--eps", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lam", type=float, default=300.0)
    ap.add_argument("--n_iter", type=int, default=15)
    ap.add_argument("--fit_reps", type=int, default=3)
    ap.add_argument("--lp", type=int, default=1, help="1: also time the optimal-channel programs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    log = _common.log

    out = dict(meta=dict(eps=args.eps, rank=args.rank, lam=args.lam, n_iter=args.n_iter,
                         machine=platform.machine(), python=platform.python_version(),
                         numpy=np.__version__, cv_fits=5 * 2 * 11),
               blocks=[], programs=[])
    for spec in args.blocks.split(","):
        ds, size = spec.split(":")
        size = int(size)
        d = args.ml_dir if ds == "ml-25m" else args.am_dir
        rows, cols, vals, shape, r_min, r_max = load_cached(ds, d, size, size, 0, d)
        grid = CH.grid_for(r_min, r_max, grid_step(ds))
        tr, _ = train_test_split(int(vals.shape[0]), 0.10, 0)
        r_tr, c_tr, v_tr = rows[tr], cols[tr], vals[tr]

        t_rule, mech = timeit(lambda: CH.select_mechanism(CH.CANDIDATES, grid, args.eps,
                                                          r_min, r_max), 200)

        def pythia():
            idx = np.abs(v_tr[:, None] - grid[None, :]).argmin(axis=1)
            counts = np.bincount(idx, minlength=len(grid)).astype(float)
            noisy = np.maximum(counts + np.random.default_rng(1).laplace(
                0.0, 2.0 / (0.1 * args.eps), size=len(grid)), 0.0)
            pi_hat = noisy / noisy.sum()
            return CH.select_mechanism(CH.CANDIDATES, grid, 0.9 * args.eps, r_min, r_max, pi=pi_hat)

        t_pythia, _ = timeit(pythia, 20)
        alpha, V, s2, cc = CH.channel(mech, grid, args.eps, r_min, r_max)
        t_release, y = timeit(lambda: CH.sample(mech, v_tr, grid, args.eps, r_min, r_max,
                                                release_rng(0, args.eps, mech)), 5)
        t_corr, x = timeit(lambda: (y - cc) / alpha, 20)
        t_fit, _ = timeit(lambda: fit_biased_mf(r_tr, c_tr, x, shape, rank=args.rank,
                                                reg=args.lam, n_iter=args.n_iter,
                                                clip=(r_min, r_max)), args.fit_reps)
        rec = dict(dataset=ds, size=size, shape=list(shape), n_train=int(v_tr.size),
                   mech=mech, rule_s=t_rule, pythia_s=t_pythia, release_s=t_release,
                   correction_s=t_corr, fit_s=t_fit)
        out["blocks"].append(rec)
        log("[time] %s %s |Omega_train|=%d mech=%s rule=%.2e s pythia=%.2e s release=%.3f s "
            "correction=%.4f s fit=%.2f s" % (ds, shape, v_tr.size, mech, t_rule, t_pythia,
                                              t_release, t_corr, t_fit))
        save(args.out, out)

    if args.lp:
        for tag, (k, lo, hi) in (("ml", (10, 0.5, 5.0)), ("am", (5, 1.0, 5.0))):
            g = np.linspace(lo, hi, k)
            D = hi - lo
            for e in (0.1, 0.25, 0.5, 1, 2, 4, 8):
                extra = np.concatenate([CF.krr_atoms(k, D, e, lo, hi), CF.geom_atoms(k, D, e, lo, hi),
                                        CF.onebit_atoms(k, D, e, lo, hi),
                                        np.linspace(lo - D / 2, hi + D / 2, 361)])
                t_lp, _ = timeit(lambda: optimal.opt_unbiased(g, e, L=161, extra=extra), 1)
                t_rr, _ = timeit(lambda: optimal.rr_bins(g, e), 1)
                out["programs"].append(dict(grid=tag, eps=e, lp_s=t_lp, rrbins_s=t_rr))
                log("[time] %s eps=%-5g LP=%.2f s RR-on-bins=%.3f s" % (tag, e, t_lp, t_rr))
                save(args.out, out)
    save(args.out, out)
    log("[done] %s" % args.out)


if __name__ == "__main__":
    main()
