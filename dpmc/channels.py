"""Entrywise eps-DP mechanisms on a finite rating grid and their public channel statistics.

A mechanism ``Q`` maps a rating ``x`` on the public grid ``g_1 < ... < g_k`` (spacing
``Delta / (k - 1)``, range ``[lo, hi]``) to a release ``Y``. For a public reference law
``pi`` on the grid the channel is summarized by

    alpha      = Cov(X, Y) / Var(X)            attenuation
    V          = E[Var(Y | X)]                 conditional noise
    sigma2_eff = V / alpha**2                  effective noise
    c          = E[Y] - alpha E[X]             intercept of the affine projection

The public affine correction ``Xtilde = (Y - c) / alpha`` is post-processing, and for a
mechanism with exactly affine conditional mean its residual variance is ``sigma2_eff``.
All four quantities depend only on the grid, the budget and ``pi``; no data is read.

Mechanism names used throughout the code base:

    krr      k-ary randomized response on the grid
    geom     truncated (folded) two-sided geometric noise on the grid index
    laplace  Laplace noise of scale Delta/eps, clipped to [lo, hi]
    shin     one-bit randomizer of Duchi et al. as deployed by Shin et al. (unbiased)
    pm       piecewise mechanism of Wang et al. (unbiased)
    hm       hybrid of one-bit and piecewise, Wang et al.
    expo     exponential mechanism with squared-error utility, calibrated exactly
    stair    staircase mechanism retuned for squared error (dpmc.staircase)
    lpopt    LP-optimal unbiased channel (dpmc.optimal)
    rrbins   randomized response on bins (dpmc.optimal)

Every function returning statistics returns the tuple ``(alpha, V, sigma2_eff, c)``.
"""
import numpy as np

from .staircase import stair_stats, stair_sample

CANDIDATES = ("krr", "geom", "laplace", "shin", "pm")
"""The short list Q over which the selection rule minimizes effective noise."""

ALL_MECHANISMS = ("krr", "geom", "laplace", "shin", "expo", "pm", "hm", "stair",
                  "lpopt", "rrbins")


def krr_matrix(k, eps):
    """Transition matrix of k-ary randomized response."""
    p = np.exp(eps) / (np.exp(eps) + k - 1.0)
    q = (1.0 - p) / (k - 1.0)
    P = np.full((k, k), q)
    np.fill_diagonal(P, p)
    return P


def geometric_matrix(k, eps):
    """Two-sided geometric noise on the grid index, folded onto the endpoints.

    The noise has pmf ``(1-a)/(1+a) a^{|z|}`` with ``a = exp(-eps/(k-1))``; the index
    sensitivity under the rating-level relation is ``k - 1`` and folding the tails onto
    the endpoints is post-processing, so the mechanism is eps-DP.
    """
    a = np.exp(-eps / (k - 1.0))
    c = (1.0 - a) / (1.0 + a)
    idx = np.arange(k)
    P = c * a ** np.abs(idx[:, None] - idx[None, :])
    for i in range(k):
        P[i, 0] += c * a ** (i + 1) / (1.0 - a)
        P[i, k - 1] += c * a ** (k - i) / (1.0 - a)
    return P


def _stats(Ey, Vcond, grid, pi):
    """Channel statistics from per-level conditional means and variances."""
    mx = float(pi @ grid)
    my = float(pi @ Ey)
    varx = float(pi @ (grid - mx) ** 2)
    alpha = float(pi @ ((grid - mx) * (Ey - my))) / varx
    V = float(pi @ Vcond)
    return alpha, V, V / alpha ** 2, my - alpha * mx


def expo_matrix(grid, eps):
    """Exponential mechanism with squared-error utility on the grid.

    ``P[i, j]`` is proportional to ``exp(-beta (g_i - g_j)^2)``. ``beta`` is found by
    bisection so that the realized privacy loss ``max_j max_{i,i'} log P[i,j]/P[i',j]``
    equals ``eps`` exactly, which is tighter than the generic calibration.
    """
    U = -(grid[:, None] - grid[None, :]) ** 2

    def mat(b):
        W = np.exp(b * U)
        return W / W.sum(1, keepdims=True)

    def loss(P):
        L = np.log(P)
        return float((L[:, None, :] - L[None, :, :]).max())

    lo_b, hi_b = 1e-12, 1e-4
    while loss(mat(hi_b)) < eps and hi_b < 1e8:
        hi_b *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo_b + hi_b)
        if loss(mat(mid)) < eps:
            lo_b = mid
        else:
            hi_b = mid
    return mat(0.5 * (lo_b + hi_b))


def expo_stats(grid, eps, pi=None):
    """Channel statistics of the exponential mechanism."""
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    return stats_from_matrix(expo_matrix(grid, eps), grid, pi)


def stats_from_matrix(P, grid, pi):
    """Channel statistics of a mechanism whose outputs are the grid levels."""
    Ey = P @ grid
    return _stats(Ey, (P @ grid ** 2) - Ey ** 2, grid, pi)


def clip_laplace_stats(grid, eps, lo, hi, pi):
    """Closed-form statistics of ``clip(x + Lap(Delta/eps), lo, hi)``."""
    b = (hi - lo) / eps
    u = grid - lo
    v = hi - grid
    bias = 0.5 * b * (np.exp(-u / b) - np.exp(-v / b))
    m2c = 2 * b ** 2 - b * (u + b) * np.exp(-u / b) - b * (v + b) * np.exp(-v / b)
    return _stats(grid + bias, m2c - bias ** 2, grid, pi)


def shin_stats(grid, eps, lo, hi, pi=None):
    """Exact statistics of the one-bit randomizer.

    With ``t = 2(x - lo)/Delta - 1`` and ``C = coth(eps/2)``, the mechanism reports ``+C``
    with probability ``(1 + t/C)/2`` and ``-C`` otherwise, then maps back to the rating
    scale. ``E[report | t] = t``, so ``alpha = 1`` and the intercept is zero; the
    conditional variance is ``(Delta/2)^2 (C^2 - t^2)``.
    """
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    D = hi - lo
    t = 2.0 * (grid - lo) / D - 1.0
    C = (np.exp(eps) + 1.0) / (np.exp(eps) - 1.0)
    Vcond = (D / 2.0) ** 2 * (C ** 2 - t ** 2)
    V = float(pi @ Vcond)
    return 1.0, V, V, 0.0


def shin_sample(vals, eps, lo, hi, rng):
    """Sample the one-bit randomizer for an array of ratings."""
    D = hi - lo
    t = 2.0 * (vals - lo) / D - 1.0
    C = (np.exp(eps) + 1.0) / (np.exp(eps) - 1.0)
    p = np.clip(0.5 * (1.0 + t / C), 0.0, 1.0)
    out = np.where(rng.random(t.shape) < p, C, -C)
    return lo + 0.5 * (out + 1.0) * D


def channel(mech, grid, eps, lo, hi, pi=None):
    """Return ``(alpha, V, sigma2_eff, c)`` of mechanism ``mech`` at budget ``eps``.

    ``pi`` is the reference law on the grid; ``None`` means uniform, which is the public
    default used by the selection rule.
    """
    k = len(grid)
    if mech in ("lpopt", "rrbins"):
        from . import optimal
        return optimal.table_stats(mech, grid, eps, pi)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    if mech == "krr":
        return stats_from_matrix(krr_matrix(k, eps), grid, pi)
    if mech == "geom":
        return stats_from_matrix(geometric_matrix(k, eps), grid, pi)
    if mech == "laplace":
        return clip_laplace_stats(grid, eps, lo, hi, pi)
    if mech == "shin":
        return shin_stats(grid, eps, lo, hi, pi)
    if mech == "expo":
        return stats_from_matrix(expo_matrix(grid, eps), grid, pi)
    if mech == "pm":
        return pm_stats(grid, eps, lo, hi, pi)
    if mech == "hm":
        return hm_stats(grid, eps, lo, hi, pi)
    if mech == "stair":
        return stair_stats(grid, eps, lo, hi, pi)
    raise ValueError(mech)


def select_mechanism(cands, grid, eps, lo, hi, pi=None):
    """The selection rule: the candidate with the smallest effective noise. Reads no data."""
    return min(cands, key=lambda m: channel(m, grid, eps, lo, hi, pi)[2])


def sample(mech, vals, grid, eps, lo, hi, rng):
    """Privatize an array of ratings independently with mechanism ``mech``."""
    k = len(grid)
    if mech in ("lpopt", "rrbins"):
        from . import optimal
        return optimal.table_sample(mech, vals, grid, eps, rng)
    if mech == "laplace":
        b = (hi - lo) / eps
        return np.clip(vals + rng.laplace(0.0, b, size=vals.shape[0]), lo, hi)
    if mech == "shin":
        return shin_sample(vals, eps, lo, hi, rng)
    if mech == "pm":
        return pm_sample(vals, eps, lo, hi, rng)
    if mech == "hm":
        return hm_sample(vals, eps, lo, hi, rng)
    if mech == "stair":
        return stair_sample(vals, grid, eps, lo, hi, rng)
    idx = np.abs(vals[:, None] - grid[None, :]).argmin(axis=1)
    P = (krr_matrix(k, eps) if mech == "krr" else
         expo_matrix(grid, eps) if mech == "expo" else geometric_matrix(k, eps))
    cum = np.cumsum(P, axis=1)
    u = rng.random(idx.shape[0])
    out = (u[:, None] > cum[idx]).sum(axis=1)
    return grid[np.clip(out, 0, k - 1)]


def conditional_variance(mech, grid, eps, lo, hi):
    """Per-level ``Var(Y | X = g_i) / alpha^2``, the noise variance of the debiased release.

    ``alpha`` is the uniform-law attenuation used by the public correction. This is the
    function ``s^2(x)`` in the least-squares risk identity.
    """
    k = len(grid)
    alpha = channel(mech, grid, eps, lo, hi)[0]
    D = hi - lo
    t = 2.0 * (grid - lo) / D - 1.0
    if mech == "shin":
        C = (np.exp(eps) + 1.0) / (np.exp(eps) - 1.0)
        v = (D / 2.0) ** 2 * (C ** 2 - t ** 2)
    elif mech == "pm":
        v = (D / 2.0) ** 2 * pm_var_t(t, eps)
    elif mech == "hm":
        a = hm_alpha(eps)
        v = (D / 2.0) ** 2 * (a * pm_var_t(t, eps) + (1.0 - a) * duchi_var_t(t, eps))
    elif mech == "laplace":
        b = (hi - lo) / eps
        u = grid - lo
        w = hi - grid
        bias = 0.5 * b * (np.exp(-u / b) - np.exp(-w / b))
        m2c = 2 * b ** 2 - b * (u + b) * np.exp(-u / b) - b * (w + b) * np.exp(-w / b)
        v = m2c - bias ** 2
    elif mech in ("krr", "geom", "expo"):
        P = (krr_matrix(k, eps) if mech == "krr" else
             expo_matrix(grid, eps) if mech == "expo" else geometric_matrix(k, eps))
        Ey = P @ grid
        v = P @ grid ** 2 - Ey ** 2
    else:
        raise ValueError(mech)
    return np.asarray(v, float) / alpha ** 2


def _pm_C(eps):
    """Output half-width ``C`` of the piecewise mechanism on ``[-1, 1]``."""
    e2 = np.exp(eps / 2.0)
    return (e2 + 1.0) / (e2 - 1.0)


def pm_var_t(t, eps):
    """Variance of the piecewise mechanism's unbiased report at input ``t`` in ``[-1, 1]``."""
    e2 = np.exp(eps / 2.0)
    return t ** 2 / (e2 - 1.0) + (e2 + 3.0) / (3.0 * (e2 - 1.0) ** 2)


def duchi_var_t(t, eps):
    """Variance of the one-bit report at input ``t`` in ``[-1, 1]``."""
    C = (np.exp(eps) + 1.0) / (np.exp(eps) - 1.0)
    return C ** 2 - t ** 2


def hm_eps_star():
    """Switching budget of the hybrid: ``log(tau)``, ``tau`` the real root of ``x^3 = x^2 + x + 1``."""
    lo, hi = 1.0, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid ** 3 - mid ** 2 - mid - 1.0 < 0.0:
            lo = mid
        else:
            hi = mid
    return float(np.log(0.5 * (lo + hi)))


def hm_alpha(eps):
    """Probability with which the hybrid uses the piecewise branch."""
    return 1.0 - np.exp(-eps / 2.0) if eps > hm_eps_star() else 0.0


def pm_stats(grid, eps, lo, hi, pi=None):
    """Exact statistics of the piecewise mechanism (unbiased, ``alpha = 1``)."""
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    D = hi - lo
    t = 2.0 * (grid - lo) / D - 1.0
    V = float(pi @ ((D / 2.0) ** 2 * pm_var_t(t, eps)))
    return 1.0, V, V, 0.0


def hm_stats(grid, eps, lo, hi, pi=None):
    """Exact statistics of the hybrid mechanism (unbiased, ``alpha = 1``)."""
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    D = hi - lo
    t = 2.0 * (grid - lo) / D - 1.0
    a = hm_alpha(eps)
    vt = a * pm_var_t(t, eps) + (1.0 - a) * duchi_var_t(t, eps)
    V = float(pi @ ((D / 2.0) ** 2 * vt))
    return 1.0, V, V, 0.0


def pm_sample_t(t, eps, rng):
    """Sample the piecewise mechanism on ``[-1, 1]``."""
    e2 = np.exp(eps / 2.0)
    C = (e2 + 1.0) / (e2 - 1.0)
    l = 0.5 * ((C + 1.0) * t - (C - 1.0))
    r = l + C - 1.0
    mid = rng.random(t.shape) < e2 / (e2 + 1.0)
    u = rng.random(t.shape)
    inner = l + u * (C - 1.0)
    left_len = l + C
    tail = np.where(u * (C + 1.0) < left_len,
                    -C + u * (C + 1.0),
                    r + (u * (C + 1.0) - left_len))
    return np.where(mid, inner, tail)


def pm_sample(vals, eps, lo, hi, rng):
    """Sample the piecewise mechanism on the rating scale."""
    D = hi - lo
    t = 2.0 * (vals - lo) / D - 1.0
    return lo + 0.5 * (pm_sample_t(t, eps, rng) + 1.0) * D


def hm_sample(vals, eps, lo, hi, rng):
    """Sample the hybrid mechanism on the rating scale."""
    D = hi - lo
    t = 2.0 * (vals - lo) / D - 1.0
    a = hm_alpha(eps)
    use_pm = rng.random(t.shape) < a
    out_pm = pm_sample_t(t, eps, rng)
    Cd = (np.exp(eps) + 1.0) / (np.exp(eps) - 1.0)
    p = np.clip(0.5 * (1.0 + t / Cd), 0.0, 1.0)
    out_d = np.where(rng.random(t.shape) < p, Cd, -Cd)
    return lo + 0.5 * (np.where(use_pm, out_pm, out_d) + 1.0) * D


def grid_for(lo, hi, step):
    """The public rating grid ``lo, lo + step, ..., hi``."""
    return np.arange(lo, hi + step / 2.0, step)
