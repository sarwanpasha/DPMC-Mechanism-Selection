"""End-to-end comparison of the LP-optimal channel and RR-on-bins with the rule's choice.

Both channels are run through the identical pipeline (``run_dpmc.py --mech lpopt`` and
``--mech rrbins``); every comparison is a paired t-test over seeds against the mechanism
the rule returns at the same budget.
"""
import argparse

import _common as C
from dpmc.stats import paired_test


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="result directories holding the fixed-mechanism and lpopt/rrbins runs")
    args = ap.parse_args()
    recs = C.load(args.runs)
    for new in ("lpopt", "rrbins"):
        ties = ahead = behind = 0
        worst_ahead = worst_behind = 0.0
        for ds in ("ml-25m", "amazon"):
            for e in C.EPS:
                r = C.rule(ds, e)
                a, _ = C.per_seed(recs, ds, new, e)
                b, _ = C.per_seed(recs, ds, r, e)
                d, hw, p = paired_test(a, b)
                base = b.mean()
                verdict = "tie"
                if p == p and p < 0.05:
                    verdict = "%s ahead" % C.NAMES[new] if d < 0 else "rule ahead"
                if verdict == "tie":
                    ties += 1
                elif d < 0:
                    ahead += 1; worst_ahead = max(worst_ahead, -d)
                else:
                    behind += 1; worst_behind = max(worst_behind, d)
                print("%-7s %-7s eps=%-5g rule=%-10s rule=%.4f new=%.4f d=%+.4f (%.2f%%) p=%s %s"
                      % (C.NAMES[new], ds, e, C.NAMES[r], base, a.mean(), d, 100 * d / base,
                         "%.4f" % p if p == p else "--", verdict))
        print("%s: ties %d, ahead %d (largest %.4f), behind %d (largest %.4f)\n"
              % (C.NAMES[new], ties, ahead, worst_ahead, behind, worst_behind))


if __name__ == "__main__":
    main()
