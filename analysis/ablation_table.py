"""The three choices made before any budget is spent, each isolated at a matched budget.

* correction: k-RR fitted to the raw release against the corrected release;
* reconstruction: nuclear-norm completion of the k-RR release at eps = 0.1, against the
  factorization of the same release and against DPMC at that budget, the nuclear-norm
  weight being the best of the fixed values tried as judged on the true held-out ratings
  (an advantage given to the ablation);
* mechanism: the spread between the best and the worst candidate at each budget.
"""
import argparse
import glob
import json

import numpy as np

import _common as C


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nodebias", nargs="+", required=True)
    ap.add_argument("--nuclear", nargs="*", default=[])
    args = ap.parse_args()
    recs = C.load(args.runs)
    raw = C.load(args.nodebias)
    for ds in ("ml-25m", "amazon"):
        print("== %s ==" % C.DATASETS[ds][0])
        for e in C.EPS:
            a = C.per_seed(raw, ds, "krr", e)[0]
            b = C.per_seed(recs, ds, "krr", e)[0]
            if a.size and b.size:
                print("correction (k-RR raw - corrected) eps=%-5g %.4f - %.4f = %+.4f"
                      % (e, a.mean(), b.mean(), a.mean() - b.mean()))
        for e in C.EPS:
            means = {m: C.per_seed(recs, ds, m, e)[0].mean() for m in C.TABLE_MECHS}
            hi, lo = max(means, key=means.get), min(means, key=means.get)
            print("mechanism spread eps=%-5g best %s %.4f, worst %s %.4f, spread %.4f"
                  % (e, C.NAMES[lo], means[lo], C.NAMES[hi], means[hi], means[hi] - means[lo]))
        nn = []
        for pattern in args.nuclear:
            for f in glob.glob(pattern):
                d = json.load(open(f))
                if d["meta"]["dataset"] == ds:
                    v = np.array([r["rmse"] for r in d["records"]])
                    nn.append((v.mean(), d["meta"].get("lam_fixed"), d["meta"]["eps_list"][0], d["records"][0]["mech"]))
        if nn:
            best = min(nn)
            mf = C.per_seed(recs, ds, best[3], best[2])[0].mean()
            print("reconstruction at eps=%g on the %s release: nuclear norm %.4f (best weight %g of %d) "
                  "vs factorization %.4f, difference %+.4f"
                  % (best[2], C.NAMES[best[3]], best[0], best[1], len(nn), mf, best[0] - mf))
            rule = C.rule(ds, best[2])
            print("DPMC at eps=%g (rule selects %s): %.4f"
                  % (best[2], C.NAMES[rule], C.per_seed(recs, ds, rule, best[2])[0].mean()))
        print()


if __name__ == "__main__":
    main()
