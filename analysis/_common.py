"""Shared helpers for the analysis scripts: result loading, the rule, display names."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dpmc import channels as CH
from dpmc.results import mechanism_records

EPS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
DATASETS = {"ml-25m": ("MovieLens-25M", 0.5, 5.0, 0.5),
            "ml-25m-rand": ("MovieLens-25M, random users", 0.5, 5.0, 0.5),
            "amazon": ("Amazon", 1.0, 5.0, 1.0),
            "goodreads": ("Goodreads", 1.0, 5.0, 1.0)}
TABLE_MECHS = ["shin", "pm", "hm", "geom", "laplace", "huber", "sw", "expo", "krr"]
"""Mechanisms with exactly or approximately affine conditional mean, compared by the rule."""
NAMES = {"shin": "one-bit", "pm": "piecewise", "hm": "hybrid", "geom": "geometric",
         "laplace": "Laplace", "huber": "Huber", "sw": "Square Wave", "expo": "exponential",
         "krr": "k-RR", "stair": "staircase", "lpopt": "LP-optimal", "rrbins": "RR-on-bins"}
METRICS = ("rmse", "recall", "ndcg")


def grid(ds):
    """``(grid, lo, hi)`` of a dataset."""
    _, lo, hi, step = DATASETS[ds]
    return CH.grid_for(lo, hi, step), lo, hi


def rule(ds, eps):
    """Mechanism returned by the data-free rule."""
    g, lo, hi = grid(ds)
    return CH.select_mechanism(CH.CANDIDATES, g, eps, lo, hi)


def per_seed(recs, ds, mech, eps, metric="rmse"):
    """Per-seed ``metric`` of ``mech`` at ``eps``, ordered by seed.

    Below its switching budget the hybrid is the one-bit randomizer (``channels.canonical``)
    and its records are those of the one-bit runs.
    """
    mech = CH.canonical(mech, eps)
    keys = sorted(k for k in recs if k[0] == ds and k[1] == mech and abs(k[2] - eps) < 1e-9)
    return np.array([recs[k][metric] for k in keys], float), [k[3] for k in keys]


def datasets_in(recs):
    """Datasets present in the loaded records, in the order of ``DATASETS``."""
    have = {k[0] for k in recs}
    return [d for d in DATASETS if d in have]


def load(paths, debias=1):
    """All DPMC records found under ``paths`` (``debias=0``: the raw-release ablation)."""
    return mechanism_records(paths, debias)
