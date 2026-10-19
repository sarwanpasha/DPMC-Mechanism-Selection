"""Read per-seed records written by the experiment drivers."""
import glob
import json
import os


def iter_result_files(paths):
    """Yield parsed JSON payloads from files and directories."""
    for p in paths:
        files = (sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
                 if os.path.isdir(p) else [p])
        for f in files:
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and "records" in d and "meta" in d:
                yield f, d


def mechanism_records(paths, debias=1):
    """Map ``(dataset, mech, eps, seed)`` to the record of a DPMC run.

    Only runs whose ``meta.debias`` equals ``debias`` are read, so the raw-release ablation
    never mixes with the corrected runs, and the budget-spending selector's files are
    skipped. The mechanism name is taken from
    ``record["mech"]`` or, failing that, from the ``"<mech>+mf"`` method tag.
    """
    out = {}
    for _, d in iter_result_files(paths):
        if "fracs" in d["meta"] or int(d["meta"].get("debias", 1)) != int(debias):
            continue
        ds = d["meta"].get("dataset")
        for r in d["records"]:
            mech = r.get("mech") or str(r.get("method", "")).replace("+mf", "")
            out[(ds, mech, round(float(r["eps"]), 4), int(r["seed"]))] = r
    return out


def hyperparameters(paths):
    """Map ``(dataset, mech, eps)`` to the ``(rank, lam)`` cross-validated for seed 0."""
    hp = {}
    for (ds, mech, eps, seed), r in sorted(mechanism_records(paths).items(), key=lambda t: -t[0][3]):
        hp[(ds, mech, eps)] = (int(r["rank"]), float(r["lam"]))
    return hp
