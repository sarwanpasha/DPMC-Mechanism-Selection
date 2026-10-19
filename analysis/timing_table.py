"""Wall-clock table from run_timing.py output."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("timing")
    args = ap.parse_args()
    d = json.load(open(args.timing))
    m = d["meta"]
    print("machine=%s python=%s numpy=%s eps=%g rank=%d lam=%g iterations=%d; CV performs %d fits"
          % (m["machine"], m["python"], m["numpy"], m["eps"], m["rank"], m["lam"], m["n_iter"], m["cv_fits"]))
    print("%-8s %-12s %10s %12s %12s %12s %10s" % ("data", "block", "|Omega|", "rule (ms)",
                                                    "Pythia (ms)", "release (s)", "fit (s)"))
    for b in d["blocks"]:
        print("%-8s %-12s %10d %12.3f %12.3f %12.3f %10.1f"
              % (b["dataset"], "%dx%d" % tuple(b["shape"]), b["n_train"], 1e3 * b["rule_s"],
                 1e3 * b["pythia_s"], b["release_s"], b["fit_s"]))
    if d.get("programs"):
        for g in ("ml", "am"):
            p = [x for x in d["programs"] if x["grid"] == g]
            if p:
                print("%s grid: LP %.2f-%.2f s, RR-on-bins %.3f-%.3f s per budget"
                      % (g, min(x["lp_s"] for x in p), max(x["lp_s"] for x in p),
                         min(x["rrbins_s"] for x in p), max(x["rrbins_s"] for x in p)))


if __name__ == "__main__":
    main()
