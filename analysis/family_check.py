"""Certified check of the full-family frontier (Proposition on the full family, part d).

For every number of levels k in a range, and every mechanism outside {one-bit, piecewise,
k-RR} with a closed or numerical effective noise, the script verifies on a geometric grid of
budgets e_1 < ... < e_N that

    sigma2_eff(M, e_{j+1}) > min{one-bit, piecewise, k-RR}(e_j)   for every j,

after checking that sigma2_eff(M, .) is strictly decreasing on the grid. Since every
effective noise decreases with the budget, the inequality on each cell [e_j, e_{j+1}]
implies that M never attains the minimum anywhere in [e_1, e_N]. It also prints the closed
crossover of the k = 3 grid, ln(1 + sqrt(6)). Effective noise is scale invariant, so the
grid is taken on [0, 1].
"""
import argparse
import math

import numpy as np

import _common  # noqa: F401  (puts the repository on the path)
from dpmc import channels as CH


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=50)
    ap.add_argument("--kmax_expo", type=int, default=20)
    ap.add_argument("--eps_lo", type=float, default=1e-2)
    ap.add_argument("--eps_hi", type=float, default=20.0)
    ap.add_argument("--points", type=int, default=8000)
    args = ap.parse_args()
    eps = np.exp(np.linspace(math.log(args.eps_lo), math.log(args.eps_hi), args.points))
    worst = {}
    ok = True
    for k in range(args.kmin, args.kmax + 1):
        g = np.linspace(0.0, 1.0, k)

        def s2(m):
            return np.array([CH.channel(m, g, e, 0.0, 1.0)[2] for e in eps])

        base = np.vstack([s2("krr"), s2("shin"), s2("pm")]).min(axis=0)
        for m in ("geom", "laplace", "sw", "expo"):
            if m == "expo" and k > args.kmax_expo:
                continue
            v = s2(m)
            decreasing = bool(np.all(np.diff(v) < 0))
            margin = float((v[1:] / base[:-1]).min())
            worst[m] = min(worst.get(m, np.inf), margin)
            if not decreasing or margin <= 1.0:
                ok = False
                print("k=%d %s: decreasing=%s smallest margin=%.6f" % (k, m, decreasing, margin))
    for m, v in worst.items():
        print("%-8s smallest cell margin over k=%d..%d: %.6f" % (m, args.kmin, args.kmax, v))
    print("certified on [%g, %g]: %s" % (args.eps_lo, args.eps_hi, ok))
    print("k=3 switch from one-bit to k-RR at ln(1 + sqrt 6) = %.6f" % math.log(1.0 + math.sqrt(6.0)))


if __name__ == "__main__":
    main()
