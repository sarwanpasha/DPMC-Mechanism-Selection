"""The budget-spending selector against the data-free rule, budget by budget.

Reads ``run_pythia.py`` outputs and the fixed-mechanism runs. For every dataset, budget
and fraction ``f``: how often the selector picks the rule's mechanism, the mean test RMSE
of both, and the paired t-test over seeds.
"""
import argparse
import glob
import json
import os

import numpy as np

import _common as C
from dpmc.stats import paired_test


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pythia", nargs="+", required=True, help="run_pythia.py result files or dirs")
    ap.add_argument("--runs", nargs="+", required=True, help="fixed-mechanism result dirs")
    args = ap.parse_args()
    recs = C.load(args.runs)
    rows = []
    for p in args.pythia:
        files = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)) if os.path.isdir(p) else [p]
        for f in files:
            d = json.load(open(f))
            ds = d["meta"]["dataset"]
            for r in d["records"]:
                rows.append((ds, round(r["eps"], 4), round(r["frac"], 4), r["seed"], r["mech"], r["rmse"]))
    summary = {}
    for ds in C.datasets_in(recs):
        for f in sorted({r[2] for r in rows}):
            for e in C.EPS:
                sel = sorted([r for r in rows if r[0] == ds and r[2] == f and abs(r[1] - e) < 1e-9],
                             key=lambda r: r[3])
                if not sel:
                    continue
                rule = C.rule(ds, e)
                a = np.array([r[5] for r in sel])
                b, _ = C.per_seed(recs, ds, rule, e)
                d, hw, p = paired_test(a, b)
                same = sum(r[4] == rule for r in sel)
                key = "rule ahead" if (p == p and p < 0.05 and d > 0) else (
                      "selector ahead" if (p == p and p < 0.05 and d < 0) else "tie")
                summary.setdefault(f, []).append((key, d, ds, e))
                print("%-7s f=%-5g eps=%-5g picks rule's mechanism %d/%d  rule=%.4f selector=%.4f d=%+.4f p=%s %s"
                      % (ds, f, e, same, len(sel), b.mean(), a.mean(), d,
                         "%.4f" % p if p == p else "--", key))
    for f, lst in sorted(summary.items()):
        n = len(lst)
        ra = sum(k == "rule ahead" for k, _, _, _ in lst)
        sa = sum(k == "selector ahead" for k, _, _, _ in lst)
        big = max(lst, key=lambda t: t[1])
        print("f=%g: rule significantly ahead at %d of %d, selector significantly ahead at %d; "
              "largest mean gap %+.4f (%s eps=%g)" % (f, ra, n, sa, big[1], big[2], big[3]))


if __name__ == "__main__":
    main()
