"""How closely the measured ordering follows the effective-noise ordering.

Reports, per dataset: the number of same-budget mechanism pairs ordered the same way by
effective noise and by test RMSE, beside two baselines, a coin flip and the best
budget-independent order (the fixed ranking of mechanisms by their average within-budget
rank, chosen in hindsight on the test results); the Spearman correlation between log
effective noise and test RMSE, within each budget and pooled; the per-budget comparison of
the rule with the mechanism the test set would have chosen (paired t-test over seeds, 95%
interval); the paired brackets of the two exchanges (one-bit vs piecewise, piecewise vs
k-RR); and the comparison of the rule with each single mechanism held fixed at every
budget. The Huber mechanism is left out of the pairwise counts because its tuned width
makes it numerically the Laplace mechanism.
"""
import argparse
import math

import numpy as np

import _common as C
from dpmc import channels as CH
from dpmc.stats import paired_test, spearman

PAIR_MECHS = ["shin", "pm", "hm", "geom", "laplace", "sw", "expo", "krr"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--metric", choices=C.METRICS, default="rmse")
    args = ap.parse_args()
    recs = C.load(args.runs)
    M = args.metric
    sign = 1.0 if M == "rmse" else -1.0
    totals = {}
    for ds in C.datasets_in(recs):
        g, lo, hi = C.grid(ds)
        print("== %s (%s) ==" % (C.DATASETS[ds][0], M))
        cells = {}
        for m in PAIR_MECHS:
            for e in C.EPS:
                a, _ = C.per_seed(recs, ds, m, e, M)
                if a.size:
                    cells[(m, e)] = (sign * a.mean(), CH.channel(CH.canonical(m, e), g, e, lo, hi)[2])
        keys = sorted(cells)
        rho = spearman([math.log(cells[k][1]) for k in keys], [cells[k][0] for k in keys])
        within = []
        for e in C.EPS:
            ms = [m for m in PAIR_MECHS if (m, e) in cells and CH.canonical(m, e) == m]
            if len(ms) > 2:
                within.append(spearman([cells[(m, e)][1] for m in ms], [cells[(m, e)][0] for m in ms]))
        avg_rank = {}
        for e in C.EPS:
            ms = [m for m in PAIR_MECHS if (m, e) in cells]
            order = sorted(ms, key=lambda m: cells[(m, e)][0])
            for r, m in enumerate(order):
                avg_rank.setdefault(m, []).append(r)
        fixed = {m: float(np.mean(v)) for m, v in avg_rank.items()}
        agree = agree_fixed = tot = 0
        for e in C.EPS:
            ms = [m for m in PAIR_MECHS if (m, e) in cells]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    r1, s1 = cells[(ms[i], e)]
                    r2, s2 = cells[(ms[j], e)]
                    if s1 == s2 or r1 == r2:
                        continue
                    tot += 1
                    agree += (s1 < s2) == (r1 < r2)
                    agree_fixed += (fixed[ms[i]] < fixed[ms[j]]) == (r1 < r2)
        print("same-budget pairs: effective noise agrees %d of %d; best fixed order %d; coin flip %.1f"
              % (agree, tot, agree_fixed, tot / 2.0))
        print("spearman(log sigma2_eff, %s): pooled %.4f, within budgets mean %.4f min %.4f"
              % (M, rho, float(np.mean(within)), float(np.min(within))))
        exact = ok = 0
        worst = 0.0
        print("%-6s %-12s %-12s %-22s %s" % ("eps", "rule", "best", "d (95%)", "p"))
        for e in C.EPS:
            r = C.rule(ds, e)
            means = {m: cells[(m, e)][0] for m in PAIR_MECHS if (m, e) in cells}
            b = CH.canonical(min(means, key=means.get), e)
            if b == r:
                exact += 1; ok += 1
                print("%-6g %-12s %-12s %-22s %s" % (e, C.NAMES[r], C.NAMES[b], "0", "--"))
                continue
            d, hw, p = paired_test(C.per_seed(recs, ds, r, e, M)[0], C.per_seed(recs, ds, b, e, M)[0])
            d *= sign
            ok += p >= 0.05
            worst = max(worst, d)
            print("%-6g %-12s %-12s %+.4f+-%.4f%8s %.3f" % (e, C.NAMES[r], C.NAMES[b], d, hw, "", p))
        print("exact agreement %d/7; rule not significantly worse at 5%%: %d/7; largest cost %.4f"
              % (exact, ok, worst))
        for a, b in (("shin", "pm"), ("pm", "krr")):
            line = []
            for e in C.EPS:
                d, hw, p = paired_test(C.per_seed(recs, ds, a, e, M)[0], C.per_seed(recs, ds, b, e, M)[0])
                d *= sign
                tag = "%s leads" % C.NAMES[a] if (p < 0.05 and d < 0) else (
                      "%s leads" % C.NAMES[b] if (p < 0.05 and d > 0) else "tie")
                line.append("%g:%s(p=%.3f)" % (e, tag, p if p == p else 1.0))
            print("%s vs %s: %s" % (C.NAMES[a], C.NAMES[b], "  ".join(line)))
        for fx in ("pm", "shin", "krr"):
            ahead = level = behind = sig_ahead = sig_behind = 0
            for e in C.EPS:
                r = C.rule(ds, e)
                d, hw, p = paired_test(C.per_seed(recs, ds, r, e, M)[0], C.per_seed(recs, ds, fx, e, M)[0])
                d *= sign
                if r == CH.canonical(fx, e) or abs(d) < 1e-12:
                    level += 1
                elif d < 0:
                    ahead += 1; sig_ahead += p < 0.05
                else:
                    behind += 1; sig_behind += p < 0.05
                print("   rule vs always-%-9s eps=%-5g rule=%-10s d=%+.4f p=%s"
                      % (C.NAMES[fx], e, C.NAMES[r], d, "%.4f" % p if p == p else "--"))
            print("rule vs always-%s: ahead %d (significant %d), level %d, behind %d (significant %d)"
                  % (C.NAMES[fx], ahead, sig_ahead, level, behind, sig_behind))
            t = totals.setdefault(fx, [0, 0, 0, 0, 0])
            for i, v in enumerate((ahead, sig_ahead, level, behind, sig_behind)):
                t[i] += v
        print()
    for fx, t in totals.items():
        print("all datasets, rule vs always-%s: ahead %d (significant %d), level %d, behind %d "
              "(significant %d)" % ((C.NAMES[fx],) + tuple(t)))


if __name__ == "__main__":
    main()
