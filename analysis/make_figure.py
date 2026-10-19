"""Figure: effective noise of the five candidates (top) and measured RMSE exchanges (bottom).

Top: closed-form effective noise against the budget on both grids, with the predicted
crossovers as dotted lines. Bottom: mean test-RMSE differences one-bit minus piecewise and
piecewise minus k-RR from the fixed-mechanism runs.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _common as C
from dpmc import closed_forms as CF

COLORS = {"krr": "#2a78d6", "pw": "#eb6834", "gm": "#1baf7a", "lp": "#eda100", "ob": "#e87ba4"}
MARK = {"krr": "o", "ob": "v"}
LABEL = {"krr": r"$k$-RR", "pw": "piecewise", "gm": "geometric", "lp": "Laplace", "ob": "one-bit"}
GRIDS = {"ml-25m": dict(k=10, D=4.5, lo=0.5, hi=5.0, ls="-"),
         "amazon": dict(k=5, D=4.0, lo=1.0, hi=5.0, ls="--")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="mechanism_crossover.pdf")
    args = ap.parse_args()
    recs = C.load(args.runs)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 13, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 9.5, "axes.linewidth": 0.9,
                         "lines.linewidth": 2.0, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 4.6))
    e = np.logspace(np.log10(0.08), np.log10(10), 240)
    for ds, g in GRIDS.items():
        k, D, lo, hi, ls = g["k"], g["D"], g["lo"], g["hi"], g["ls"]
        curves = {"krr": [CF.krr(k, D, x) for x in e], "pw": [CF.piecewise(k, D, x) for x in e],
                  "ob": [CF.onebit(k, D, x) for x in e], "gm": [CF.geometric(k, D, x, lo, hi) for x in e],
                  "lp": [CF.laplace(k, D, x, lo, hi, n=20001) for x in e]}
        for key in ("krr", "pw", "gm", "lp", "ob"):
            ax.plot(e, curves[key], ls, color=COLORS[key], label=LABEL[key] if ds == "ml-25m" else None)
        for v in CF.crossovers(k, D):
            ax.axvline(v, color="0.45", lw=0.8, ls=":")
            bx.axvline(v, color="0.45", lw=0.8, ls=":")
        eps = np.array(C.EPS)
        mean = {m: np.array([C.per_seed(recs, ds, m, x)[0].mean() for x in C.EPS]) for m in ("shin", "pm", "krr")}
        bx.plot(eps, mean["shin"] - mean["pm"], ls, color=COLORS["ob"], marker=MARK["ob"], ms=5.5,
                label=r"one-bit $-$ piecewise" if ds == "ml-25m" else None)
        bx.plot(eps, mean["pm"] - mean["krr"], ls, color=COLORS["krr"], marker=MARK["krr"], ms=5.5,
                label=r"piecewise $-$ $k$-RR" if ds == "ml-25m" else None)
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlabel(r"privacy budget $\varepsilon$")
        a.set_xticks([0.1, 0.5, 1, 2, 4, 8]); a.set_xticklabels(["0.1", "0.5", "1", "2", "4", "8"])
        a.grid(True, which="major", color="0.88", lw=0.6); a.set_axisbelow(True)
    ax.set_yscale("log"); ax.set_ylabel(r"$\sigma^2_{\mathrm{eff}}$"); ax.set_ylim(2e-3, 2e5)
    ax.legend(ncol=2, frameon=False, handlelength=1.6, columnspacing=0.9, labelspacing=0.22,
              borderpad=0.1, loc="upper right")
    bx.axhline(0, color="0.3", lw=0.9)
    bx.set_ylabel(r"$\Delta$RMSE")
    bx.legend(frameon=False, handlelength=1.8, labelspacing=0.22, borderpad=0.1, loc="upper left")
    fig.tight_layout(pad=0.3, h_pad=1.0)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
