"""Synthetic k-sweep: at how many budgets the piecewise (middle) regime is predicted and measured.

Predicted: budgets at which the piecewise mechanism minimizes effective noise on the
synthetic grid. Measured: budgets at which it has the lowest mean test RMSE among the
mechanisms run.
"""
import argparse
import json

import numpy as np

from _common import EPS
from dpmc import channels as CH


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for f in sorted(args.files, key=lambda p: json.load(open(p))["meta"]["k"]):
        d = json.load(open(f))
        k = d["meta"]["k"]
        grid = np.array(d["grid"])
        mechs = sorted({r["mech"] for r in d["records"]})
        pred = meas = 0
        for e in EPS:
            s = {m: CH.channel(m, grid, e, 1.0, 5.0)[2] for m in ("krr", "geom", "laplace", "shin", "pm")}
            pred += min(s, key=s.get) == "pm"
            means = {m: np.mean([r["rmse"] for r in d["records"] if r["mech"] == m and r["eps"] == e])
                     for m in mechs}
            meas += min(means, key=means.get) == "pm"
        print("k=%-3d piecewise regime: predicted at %d budgets, measured best at %d budgets "
              "(non-private %.4f)" % (k, pred, meas, d["nonprivate"]))


if __name__ == "__main__":
    main()
