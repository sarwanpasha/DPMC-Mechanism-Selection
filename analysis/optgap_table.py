"""What the candidate list costs against the exact optimum, and what RR-on-bins does.

For each grid and budget: the effective noise of the rule's choice, the minimum over all
eps-DP channels with affine conditional mean (the LP of dpmc.optimal, read from the
tabulated optimal channel), the relative excess, and for RR-on-bins its effective noise,
attenuation and the residual ``E[r^2] / alpha^2`` of the affine correction. Also reports
the sparsity of the LP solution and the number of distinct likelihoods per output symbol.
"""
import numpy as np

import _common as C
from dpmc import channels as CH, optimal


def main():
    for ds in ("ml-25m", "amazon"):
        g, lo, hi = C.grid(ds)
        print("== %s grid, k=%d ==" % (C.DATASETS[ds][0], len(g)))
        print("%-6s %10s %10s %8s | %10s %7s %8s | %5s %s"
              % ("eps", "rule", "exact", "excess", "RRbins", "alpha", "Er2/a2", "atoms", "likelihoods/symbol"))
        for e in C.EPS:
            r = C.rule(ds, e)
            s_rule = CH.channel(r, g, e, lo, hi)[2]
            z, P = optimal.table("lpopt", g, e)
            s_lp = optimal.table_stats("lpopt", g, e)[2]
            zr, Pr = optimal.table("rrbins", g, e)
            a, c, sup, r2 = optimal.affine_residual(g, zr, Pr)
            s_rr = optimal.table_stats("rrbins", g, e)[2]
            levels = [len(np.unique(np.round(P[:, j] / P[:, j].max(), 6))) for j in range(P.shape[1])]
            print("%-6g %10.4g %10.4g %7.1f%% | %10.4g %7.3f %8.3f | %5d %s"
                  % (e, s_rule, s_lp, 100 * (s_rule / s_lp - 1), s_rr, a, r2, len(z), sorted(set(levels))))
        print()


if __name__ == "__main__":
    main()
