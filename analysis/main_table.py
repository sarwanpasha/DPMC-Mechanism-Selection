"""Test RMSE of DPMC and of every mechanism held fixed, with the reference rows.

Usage: ``python main_table.py --runs <dpmc result dirs> [--metric rmse|recall|ndcg]
[--dpsgd <files>] [--dpgd <files>] [--reference <files>]``. Entries are mean and sample
standard deviation over the seeds, for every dataset present in the runs.
The DPMC row is the mechanism the rule returns at each budget, so it coincides with that
mechanism's row. The best pure-eps entry per budget (lowest RMSE, highest Recall@10 or NDCG@10) is marked with
``*``; DP-SGD carries
``(eps, delta)`` and is excluded from the marking.
"""
import argparse
import json

import numpy as np

import _common as C


def fmt(a):
    """Mean and sample standard deviation."""
    return "%.4f+-%.4f" % (a.mean(), a.std(ddof=1)) if a.size else "-"


def runs_by_eps(files):
    """Per-budget per-seed RMSE from baseline result files (DP-SGD / DP-GD)."""
    out = {}
    for f in files or []:
        d = json.load(open(f))
        ds = d["meta"]["dataset"]
        for r in d["records"]:
            out.setdefault((ds, round(float(r["eps"]), 4)), []).append((int(r["seed"]), float(r["rmse"])))
    return {k: np.array([v for _, v in sorted(s)]) for k, s in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dpsgd", nargs="*", default=[])
    ap.add_argument("--dpgd", nargs="*", default=[])
    ap.add_argument("--reference", nargs="*", default=[])
    ap.add_argument("--metric", choices=C.METRICS, default="rmse")
    args = ap.parse_args()
    recs = C.load(args.runs)
    M = args.metric
    sign = 1.0 if M == "rmse" else -1.0
    sgd, gd = runs_by_eps(args.dpsgd), runs_by_eps(args.dpgd)
    for ds in C.datasets_in(recs):
        print("== %s (%s) ==" % (C.DATASETS[ds][0], M))
        print("%-22s" % "method" + "".join("%17g" % e for e in C.EPS))
        rows = {"DPMC (rule-selected)": [C.per_seed(recs, ds, C.rule(ds, e), e, M)[0] for e in C.EPS]}
        for m in C.TABLE_MECHS + ["stair", "lpopt", "rrbins"]:
            rows[C.NAMES[m]] = [C.per_seed(recs, ds, m, e, M)[0] for e in C.EPS]
        if M == "rmse":
            rows["DP-GD (pure eps)"] = [gd.get((ds, e), np.array([])) for e in C.EPS]
        best = []
        for j in range(len(C.EPS)):
            cand = {k: sign * v[j].mean() for k, v in rows.items() if v[j].size}
            best.append(min(cand, key=cand.get) if cand else None)
        for name, vals in rows.items():
            cells = []
            for j, a in enumerate(vals):
                star = "*" if best[j] and (best[j] == name or (a.size and abs(
                    a.mean() - rows[best[j]][j].mean()) < 1e-12)) else " "
                cells.append("%16s%s" % (fmt(a), star))
            print("%-22s" % name + "".join(cells))
        if M == "rmse":
            print("%-22s" % "DP-SGD (delta=1e-6)"
                  + "".join("%17s" % fmt(sgd.get((ds, e), np.array([]))) for e in C.EPS))
        for f in args.reference:
            d = json.load(open(f))
            if d["meta"]["dataset"] == ds:
                for r in d["runs"]:
                    print("%-22s %.4f +- %.4f" % (r["method"], r["rmse_mean"],
                                                  np.std(r["rmse_per_seed"], ddof=1)))
        print()


if __name__ == "__main__":
    main()
