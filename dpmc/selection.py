"""Deterministic seeding and cross-validation on privatized values.

Every random draw in the experiments comes from a NumPy generator seeded by a tuple of
integers, so a run is a deterministic function of its arguments. The mechanism name
enters through a SHA-256 hash because Python's built-in ``hash`` of a string changes
between processes.

Hyperparameters ``(rank, lam)`` are chosen by K-fold cross-validation against the
debiased privatized values of the seed-0 release, independently at each budget. Neither
the selection nor the fit reads a true rating, so both are post-processing of the release.
"""
import hashlib
import time

import numpy as np

from .data import rmse
from .mf import fit_biased_mf


def stable_hash(s):
    """Process-independent 32-bit hash of a string."""
    return int(hashlib.sha256(str(s).encode()).hexdigest()[:8], 16)


def release_rng(seed, eps, mech):
    """Generator for the privatization draw of ``(seed, eps, mech)``."""
    return np.random.default_rng([seed, int(round(eps * 1000)), abs(stable_hash(mech)) % 1000])


def fit_eval(r_tr, c_tr, x_tr, shape, rank, lam, n_iter, clip):
    """Fit the biased factorization at fixed hyperparameters."""
    return fit_biased_mf(r_tr, c_tr, x_tr, shape, rank=rank, reg=lam,
                         n_iter=n_iter, clip=clip)


def score_table(r_tr, c_tr, x_tr, shape, ranks, lams, n_iter, clip, seed, folds, log):
    """Mean K-fold held-out RMSE on the debiased privatized values, per ``(rank, lam)``."""
    n = x_tr.shape[0]
    perm = np.random.default_rng(9000 + seed).permutation(n)
    cut = np.array_split(perm, folds)
    S = np.zeros((len(ranks), len(lams)))
    for i, rk in enumerate(ranks):
        for j, lam in enumerate(lams):
            t0, acc = time.time(), []
            for f in range(folds):
                sv = cut[f]
                st = np.concatenate([cut[g] for g in range(folds) if g != f])
                m = fit_eval(r_tr[st], c_tr[st], x_tr[st], shape, rk, lam, n_iter, clip)
                acc.append(float(rmse(m["predict"](r_tr[sv], c_tr[sv]), x_tr[sv])))
            S[i, j] = float(np.mean(acc))
            log("      [sel] rank=%3d lam=%-8g cv=%.4f (%.0fs)"
                % (rk, lam, S[i, j], time.time() - t0))
    return S


def pick(S, ranks, lams):
    """The ``(rank, lam)`` with the smallest cross-validation score."""
    i, j = np.unravel_index(int(np.argmin(S)), S.shape)
    return ranks[i], lams[j]
