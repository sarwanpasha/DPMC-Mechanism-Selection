"""Every data-free number in the paper: effective noise, crossovers, reference-law analyses.

Prints the effective-noise table of the seven candidates on both grids (smallest per
budget marked), the two crossover budgets as functions of ``k`` with the scale-invariance
check, the range of the second crossover over the exponential-tilt family, the law-free
boundaries, and the number of budgets at which the rule's choice beats five of its six
rivals under every law.
"""
import numpy as np

import _common as C
from dpmc import channels as CH, closed_forms as CF


def main():
    for ds in ("ml-25m", "amazon"):
        g, lo, hi = C.grid(ds)
        k, D = len(g), hi - lo
        print("== %s: k=%d Delta=%g sigma2_pi=%.4f ==" % (C.DATASETS[ds][0], k, D, CF.sigma2_pi(k, D)))
        tab = {m: [CH.channel(m, g, e, lo, hi)[2] for e in C.EPS] for m in C.TABLE_MECHS}
        best = [min(C.TABLE_MECHS, key=lambda m: tab[m][j]) for j in range(len(C.EPS))]
        print("%-12s" % "mechanism" + "".join("%12g" % e for e in C.EPS))
        for m in C.TABLE_MECHS:
            print("%-12s" % C.NAMES[m] + "".join(
                ("%11.4g*" if best[j] == m else "%12.4g") % tab[m][j] for j in range(len(C.EPS))))
        print("rule:", [C.rule(ds, e) for e in C.EPS])
        e1, e2 = CF.crossovers(k, D)
        print("crossovers: eps1=%.4f eps2=%.4f" % (e1, e2))
        th, ga = np.linspace(-4, 4, 41), np.linspace(-8, 8, 41)
        print("eps2 over the tilt family theta in [-4,4], gamma in [-8,8]: [%.2f, %.2f]"
              % CF.tilt_sweep(g, lo, hi, th, ga))
        print("one-bit beats piecewise under every law below eps=%.3f"
              % CF.law_free_boundary("shin", "pm", g, lo, hi, 0.1, 3.0))
        print("piecewise beats k-RR under every law below eps=%.2f; k-RR beats piecewise above eps=%.2f"
              % (CF.law_free_boundary("pm", "krr", g, lo, hi, 0.5, 6.0),
                 CF.law_free_boundary("krr", "pm", g, lo, hi, 2.0, 19.0)))
        count = 0
        for e in C.EPS:
            r = C.rule(ds, e)
            beaten = sum(CF.law_free(r, m, g, e, lo, hi) for m in C.TABLE_MECHS if m != r)
            count += beaten >= 5
            print("  eps=%-5g rule=%-6s beats %d of 6 rivals under every law" % (e, C.NAMES[r], beaten))
        print("budgets at which the rule's choice beats at least five of six rivals law-free: %d of 7" % count)
        print()
    print("== crossovers versus k (Delta=4.5; Delta=10 gives identical values) ==")
    for k in (3, 4, 5, 6, 7, 8, 10, 12, 15, 20):
        a, b = CF.crossovers(k, 4.5)
        a2, b2 = CF.crossovers(k, 10.0)
        print("k=%-3d eps1=%.4f eps2=%.4f  | Delta=10: %.4f %.4f" % (k, a, b, a2, b2))


if __name__ == "__main__":
    main()
