"""How closely the measured ordering follows the effective-noise ordering.

Reports, per dataset: the Spearman correlation between log effective noise and test RMSE
over all (mechanism, budget) cells; the number of same-budget mechanism pairs ordered the
same way by both; the per-budget comparison of the rule with the mechanism the test set
would have chosen (paired t-test over seeds, 95% interval); the paired brackets of the two
exchanges (one-bit vs piecewise, piecewise vs k-RR); and the comparison of the rule with
each single mechanism held fixed at every budget.
"""
import argparse
import math

import numpy as np

import _common as C
from dpmc import channels as CH
from dpmc.stats import paired_test, spearman


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    recs = C.load(args.runs)
    totals = {}
    for ds in ("ml-25m", "amazon"):
        g, lo, hi = C.grid(ds)
        print("== %s ==" % C.DATASETS[ds][0])
        cells = {}
        for m in C.TABLE_MECHS:
            for e in C.EPS:
                a, _ = C.per_seed(recs, ds, m, e)
                if a.size:
                    cells[(m, e)] = (a.mean(), CH.channel(m, g, e, lo, hi)[2])
        keys = sorted(cells)
        rho = spearman([math.log(cells[k][1]) for k in keys], [cells[k][0] for k in keys])
        agree = tot = 0
        for e in C.EPS:
            ms = [m for m in C.TABLE_MECHS if (m, e) in cells]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    r1, s1 = cells[(ms[i], e)]
                    r2, s2 = cells[(ms[j], e)]
                    if s1 == s2 or r1 == r2:
                        continue
                    tot += 1
                    agree += (s1 < s2) == (r1 < r2)
        print("cells=%d spearman(log sigma2_eff, RMSE)=%.4f same-budget agreement=%d of %d"
              % (len(cells), rho, agree, tot))
        exact = ok = 0
        worst = 0.0
        print("%-6s %-12s %-12s %-22s %s" % ("eps", "rule", "best", "dRMSE (95%)", "p"))
        for e in C.EPS:
            r = C.rule(ds, e)
            means = {m: cells[(m, e)][0] for m in C.TABLE_MECHS if (m, e) in cells}
            b = min(means, key=means.get)
            if b == r:
                exact += 1; ok += 1
                print("%-6g %-12s %-12s %-22s %s" % (e, C.NAMES[r], C.NAMES[b], "0", "--"))
                continue
            d, hw, p = paired_test(C.per_seed(recs, ds, r, e)[0], C.per_seed(recs, ds, b, e)[0])
            ok += p >= 0.05
            worst = max(worst, d)
            print("%-6g %-12s %-12s %+.4f+-%.4f%8s %.3f" % (e, C.NAMES[r], C.NAMES[b], d, hw, "", p))
        print("exact agreement %d/7; rule not significantly worse at 5%%: %d/7; largest cost %.4f"
              % (exact, ok, worst))
        for a, b in (("shin", "pm"), ("pm", "krr")):
            line = []
            for e in C.EPS:
                d, hw, p = paired_test(C.per_seed(recs, ds, a, e)[0], C.per_seed(recs, ds, b, e)[0])
                tag = "%s leads" % C.NAMES[a] if (p < 0.05 and d < 0) else (
                      "%s leads" % C.NAMES[b] if (p < 0.05 and d > 0) else "tie")
                line.append("%g:%s(p=%.3f)" % (e, tag, p if p == p else 1.0))
            print("%s vs %s: %s" % (C.NAMES[a], C.NAMES[b], "  ".join(line)))
        for fixed in ("pm", "shin"):
            ahead = level = behind = sig_ahead = sig_behind = 0
            pa = []
            for e in C.EPS:
                r = C.rule(ds, e)
                d, hw, p = paired_test(C.per_seed(recs, ds, r, e)[0], C.per_seed(recs, ds, fixed, e)[0])
                if r == fixed or abs(d) < 1e-12:
                    level += 1
                elif d < 0:
                    ahead += 1; sig_ahead += p < 0.05; pa.append(p)
                else:
                    behind += 1; sig_behind += p < 0.05
                print("   rule vs always-%-6s eps=%-5g rule=%-10s d=%+.4f p=%s"
                      % (C.NAMES[fixed], e, C.NAMES[r], d, "%.4f" % p if p == p else "--"))
            print("rule vs always-%s: ahead %d (significant %d), level %d, behind %d (significant %d)"
                  % (C.NAMES[fixed], ahead, sig_ahead, level, behind, sig_behind))
            t = totals.setdefault(fixed, [0, 0, 0, 0, 0])
            for i, v in enumerate((ahead, sig_ahead, level, behind, sig_behind)):
                t[i] += v
        print()
    for fixed, t in totals.items():
        print("both datasets, rule vs always-%s: ahead %d (significant %d), level %d, behind %d "
              "(significant %d)" % ((C.NAMES[fixed],) + tuple(t)))


if __name__ == "__main__":
    main()
