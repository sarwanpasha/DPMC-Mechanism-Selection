"""Staircase mechanism of Geng and Viswanath on a bounded rating scale.

The additive noise has the staircase density with step width ``Delta = hi - lo`` and
shape parameter ``gamma``. The release is clipped to ``[lo, hi]``, which is
post-processing. ``gamma`` is tuned to minimize the effective noise (squared error)
rather than absolute error, from the public grid and budget alone, so the tuning
spends no privacy budget.
"""
from functools import lru_cache

import numpy as np


def _sc_A(eps, D, g):
    """Density level on the central step."""
    return (1.0 - np.exp(-eps)) / (2.0 * D * (g + (1.0 - g) * np.exp(-eps)))


def sc_cdf(z, eps, D, g):
    """Cumulative distribution function of staircase noise at ``z``."""
    z = np.asarray(z, float)
    s = np.sign(z); az = np.abs(z)
    A = _sc_A(eps, D, g)
    kk = np.floor(az / D); r = az - kk * D
    e_k = np.exp(-eps * kk)
    part = np.where(r <= g * D, A * e_k * r,
                    A * e_k * g * D + A * e_k * np.exp(-eps) * (r - g * D))
    return 0.5 + s * (0.5 * (1.0 - e_k) + part)


def _segments(eps, D, g):
    """Constant-density pieces of the noise on ``[-D, D]``."""
    A = _sc_A(eps, D, g); e = np.exp(-eps)
    segs = [(-D, -g * D, A * e), (-g * D, 0.0, A), (0.0, g * D, A), (g * D, D, A * e)]
    return [s for s in segs if s[1] > s[0]]


def sc_cond_moments(x, eps, lo, hi, g):
    """First two moments of the clipped release ``clip(x + Z, lo, hi)``."""
    D = hi - lo
    zlo, zhi = lo - x, hi - x
    p_lo = float(sc_cdf(zlo, eps, D, g)); p_hi = float(1.0 - sc_cdf(zhi, eps, D, g))
    m1 = lo * p_lo + hi * p_hi
    m2 = lo * lo * p_lo + hi * hi * p_hi
    for a, b, d in _segments(eps, D, g):
        a = max(a, zlo); b = min(b, zhi)
        if b <= a:
            continue
        wa, wb = x + a, x + b
        m1 += d * 0.5 * (wb ** 2 - wa ** 2)
        m2 += d * (wb ** 3 - wa ** 3) / 3.0
    return m1, m2


def staircase_stats(grid, eps, lo, hi, pi=None, g=None):
    """Channel statistics ``(alpha, V, sigma2_eff, intercept)`` of the clipped staircase."""
    if g is None:
        g = sc_best_gamma(len(grid), eps, lo, hi)
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    Ey = np.empty(k); Ey2 = np.empty(k)
    for i, x in enumerate(grid):
        Ey[i], Ey2[i] = sc_cond_moments(float(x), eps, lo, hi, g)
    V = float(pi @ (Ey2 - Ey ** 2))
    mx = float(pi @ grid); my = float(pi @ Ey)
    varx = float(pi @ (grid - mx) ** 2)
    alpha = float(pi @ ((grid - mx) * (Ey - my))) / varx
    return alpha, V, V / alpha ** 2, my - alpha * mx


@lru_cache(maxsize=None)
def sc_best_gamma(k, eps, lo, hi, n=241):
    """Shape parameter minimizing the effective noise under the uniform law."""
    grid = np.linspace(lo, hi, k)
    best, bg = np.inf, 1e-4
    for g in np.linspace(1e-4, 1.0 - 1e-4, n):
        s = staircase_stats(grid, eps, lo, hi, None, g)[2]
        if np.isfinite(s) and s < best:
            best, bg = s, g
    return float(bg)


def stair_stats(grid, eps, lo, hi, pi=None):
    """Channel statistics at the effective-noise-optimal shape parameter."""
    return staircase_stats(grid, eps, lo, hi, pi,
                           sc_best_gamma(len(grid), eps, lo, hi))


def stair_sample(vals, grid, eps, lo, hi, rng):
    """Release ``clip(x + Z, lo, hi)`` by inverse-CDF sampling of the staircase noise."""
    D = hi - lo
    g = sc_best_gamma(len(grid), eps, lo, hi)
    u = rng.random(np.shape(vals))
    a = np.full(np.shape(vals), -4.0 * D); b = np.full(np.shape(vals), 4.0 * D)
    for _ in range(64):
        m = 0.5 * (a + b)
        below = sc_cdf(m, eps, D, g) < u
        a = np.where(below, m, a); b = np.where(below, b, m)
    return np.clip(vals + 0.5 * (a + b), lo, hi)


def dp_slack(eps, D, g, n=8000):
    """Largest log density ratio over shifts ``|s| <= D`` minus ``eps``; nonpositive iff eps-DP."""
    A = _sc_A(eps, D, g)

    def dens(z):
        az = np.abs(z); kk = np.floor(az / D); r = az - kk * D
        return A * np.exp(-eps * kk) * np.where(r <= g * D, 1.0, np.exp(-eps))

    zs = np.linspace(-3 * D, 3 * D, n); worst = -np.inf
    for s in np.linspace(-D, D, 41):
        worst = max(worst, float(np.nanmax(np.log(dens(zs) / dens(zs + s)))))
    return worst - eps
