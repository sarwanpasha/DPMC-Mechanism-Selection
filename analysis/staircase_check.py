"""Does the staircase mechanism's lower effective noise translate into lower error?

For every grid and budget compares the retuned staircase with the best of the one-bit,
piecewise and k-RR mechanisms by effective noise, and runs the paired t-test on test RMSE.
"""
import argparse

import _common as C
from dpmc import channels as CH
from dpmc.stats import paired_test


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    recs = C.load(args.runs)
    lower = wins = losses = 0
    for ds in ("ml-25m", "amazon"):
        g, lo, hi = C.grid(ds)
        for e in C.EPS:
            s = {m: CH.channel(m, g, e, lo, hi)[2] for m in ("shin", "pm", "krr", "stair")}
            rival = min(("shin", "pm", "krr"), key=s.get)
            d, hw, p = paired_test(C.per_seed(recs, ds, "stair", e)[0], C.per_seed(recs, ds, rival, e)[0])
            low = s["stair"] < s[rival]
            lower += low
            if p == p and p < 0.05:
                wins += d < 0; losses += d > 0
            print("%-7s eps=%-5g stair=%.4g rival=%s %.4g lower=%s dRMSE=%+.4f p=%s"
                  % (ds, e, s["stair"], C.NAMES[rival], s[rival], low, d, "%.4f" % p if p == p else "--"))
    print("staircase has lower sigma2_eff at %d of 14 pairs; end to end it wins at %d and loses at %d"
          % (lower, wins, losses))


if __name__ == "__main__":
    main()
