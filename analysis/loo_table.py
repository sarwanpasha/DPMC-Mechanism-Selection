"""Summary of the leave-one-out design-correlation measurements (run_loo_bias.py outputs)."""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loo", nargs="+", required=True)
    args = ap.parse_args()
    allr, holds, ident = [], True, 0.0
    per = {}
    for p in args.loo:
        files = sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
        for f in files:
            d = json.load(open(f))
            if "summary" not in d:
                continue
            m = d["meta"]
            ratios = np.array([r["ratio"] for r in d["records"]])
            allr.append(ratios)
            holds &= d["summary"]["bound_holds"]
            ident = max(ident, d["summary"]["max_identity_residual"])
            per[(m["dataset"], m["mech"], m["eps"])] = (np.median(ratios), np.percentile(ratios, 95),
                                                        ratios.max())
    for k in sorted(per):
        print("%-7s %-8s eps=%-5g median=%.4f p95=%.4f max=%.4f" % (k + per[k]))
    a = np.concatenate(allr) if allr else np.array([])
    print("configurations=%d users=%d median ratio=%.4f p95=%.4f max=%.4f bound holds everywhere=%s "
          "max identity residual=%.2e" % (len(per), a.size, np.median(a), np.percentile(a, 95),
                                          a.max(), holds, ident))
    low = {k: v for k, v in per.items() if k[2] <= 1.0}
    high = {k: v for k, v in per.items() if k[2] > 1.0}
    for name, grp in (("eps <= 1", low), ("eps > 1", high)):
        if grp:
            print("%s: %d configurations, largest per-configuration median %.4f, 95th percentile "
                  "%.4f, max %.4f" % (name, len(grp), max(v[0] for v in grp.values()),
                                      max(v[1] for v in grp.values()), max(v[2] for v in grp.values())))


if __name__ == "__main__":
    main()
