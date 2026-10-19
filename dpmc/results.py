"""Read per-seed records written by the experiment drivers."""
import glob
import json
import os


def iter_result_files(paths):
    """Yield parsed JSON payloads from files and directories."""
    for p in paths:
        files = sorted(glob.glob(os.path.join(p, "*.json"))) if os.path.isdir(p) else [p]
        for f in files:
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and "records" in d and "meta" in d:
                yield f, d


def mechanism_records(paths):
    """Map ``(dataset, mech, eps, seed)`` to the record of a DPMC run.

    The mechanism name is taken from ``record["mech"]`` or, failing that, from the
    ``"<mech>+mf"`` method tag.
    """
    out = {}
    for _, d in iter_result_files(paths):
        ds = d["meta"].get("dataset")
        for r in d["records"]:
            mech = r.get("mech") or str(r.get("method", "")).replace("+mf", "")
            out[(ds, mech, round(float(r["eps"]), 4), int(r["seed"]))] = r
    return out


def hyperparameters(paths):
    """Map ``(dataset, mech, eps)`` to the cross-validated ``(rank, lam)``."""
    hp = {}
    for (ds, mech, eps, seed), r in mechanism_records(paths).items():
        hp[(ds, mech, eps)] = (int(r["rank"]), float(r["lam"]))
    return hp
