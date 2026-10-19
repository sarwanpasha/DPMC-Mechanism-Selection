"""Solve the LP lower bound and RR-on-bins on the two evaluation grids at every budget.

For each grid and budget the LP is solved on an output alphabet of 161 equally spaced
points plus the grid, the corrected k-RR and geometric atoms, the one-bit atoms and a
fine grid extending half a range beyond each end; atoms receiving no mass are dropped.
RR-on-bins is solved exactly by enumeration. The script prints the effective noise,
realized privacy loss and affine residual of both channels, and writes the transition
tables in the format of ``dpmc/optimal_tables.txt`` (12 significant digits). The file
shipped in ``dpmc/`` is the one used for the end-to-end runs; this script writes to
``--out`` so the shipped file is never overwritten by accident.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import _common
from dpmc import closed_forms as C, optimal

GRIDS = {"ml": dict(k=10, D=4.5, lo=0.5, hi=5.0), "am": dict(k=5, D=4.0, lo=1.0, hi=5.0)}
EPS = [0.1, 0.25, 0.5, 1, 2, 4, 8]


def job(task):
    """Solve both programs for one grid and budget."""
    ds, eps = task
    gd = GRIDS[ds]
    g = np.linspace(gd["lo"], gd["hi"], gd["k"])
    extra = np.concatenate([C.krr_atoms(gd["k"], gd["D"], eps, gd["lo"], gd["hi"]),
                            C.geom_atoms(gd["k"], gd["D"], eps, gd["lo"], gd["hi"]),
                            C.onebit_atoms(gd["k"], gd["D"], eps, gd["lo"], gd["hi"]),
                            np.linspace(gd["lo"] - gd["D"] / 2, gd["hi"] + gd["D"] / 2, 361)])
    val, P, z = optimal.opt_unbiased(g, eps, L=161, extra=extra)
    keep = P.max(axis=0) > 1e-10
    z1, P1 = z[keep], P[:, keep]
    P1 = P1 / P1.sum(axis=1, keepdims=True)
    zr, Pr, nb, _ = optimal.rr_bins(g, eps)
    return ds, eps, g, (z1, P1, val), (zr, Pr, nb)


def fmt(name, A):
    """One table line: name, shape and values to 12 significant digits."""
    shape = "%d,%d" % A.shape if A.ndim == 2 else "%d" % A.size
    return "%s %s %s" % (name, shape, " ".join("%.12g" % x for x in A.ravel()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="optimal_tables_regenerated.txt")
    args = ap.parse_args()
    log = _common.log
    tasks = [(ds, e) for ds in GRIDS for e in EPS]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        res = list(ex.map(job, tasks))
    lines = []
    for ds, eps, g, (z1, P1, val), (zr, Pr, nb) in sorted(res, key=lambda t: (t[0], t[1])):
        tag = "%s_%g" % (ds, eps)
        ar, cr, supr, r2r = optimal.affine_residual(g, zr, Pr)
        log("%-3s eps=%-5g | LP: value=%-11.5g loss=%.6f atoms=%-3d | RR-on-bins: bins=%d "
            "alpha=%.4f sup|r|=%.4f E[r^2]/alpha^2=%.4f loss=%.6f"
            % (ds, eps, val, optimal.privacy_loss(P1), len(z1), nb, ar, supr, r2r,
               optimal.privacy_loss(Pr)))
        lines += [fmt("lpopt_P_" + tag, P1), fmt("lpopt_z_" + tag, z1),
                  fmt("rrbins_P_" + tag, Pr), fmt("rrbins_z_" + tag, zr)]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    log("wrote %s" % args.out)


if __name__ == "__main__":
    main()
