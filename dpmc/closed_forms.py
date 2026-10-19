"""Closed-form effective noise, the two crossover budgets, and reference-law analyses.

On a grid of ``k`` equally spaced levels of width ``Delta`` under the uniform law,
``sigma2_pi = Delta^2 (k + 1) / (12 (k - 1))`` and ``E[t^2] = (k + 1) / (3 (k - 1))`` for
``t = 2(x - lo)/Delta - 1``. The effective noise of the three mechanisms that define the
selection frontier is

    k-RR       sigma2_pi (a^-2 - 1),  a = (e^eps - 1) / (e^eps + k - 1)
    one-bit    (Delta^2 / 4) (coth^2(eps/2) - E[t^2])
    piecewise  (Delta^2 / 4) (E[t^2] / (u^2 - 1) + (u^2 + 3) / (3 (u^2 - 1)^2)),  u = e^{eps/4}

The first crossover solves ``rho_1 = sigma2(one-bit) / sigma2(piecewise) = 1`` and the
second ``rho_2 = sigma2(piecewise) / sigma2(k-RR) = 1``. Both ratios are free of
``Delta``, so the crossovers depend on ``k`` alone.
"""
import numpy as np

from . import channels as CH

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def grid(k, lo, hi):
    """The public grid of ``k`` equally spaced levels on ``[lo, hi]``."""
    return np.linspace(lo, hi, k)


def sigma2_pi(k, D):
    """Variance of the uniform law on a ``k``-level grid of width ``D``."""
    return D ** 2 * (k + 1) / (12.0 * (k - 1))


def krr(k, D, eps):
    """Effective noise of k-ary randomized response."""
    a = (np.exp(eps) - 1.0) / (np.exp(eps) + k - 1.0)
    return sigma2_pi(k, D) * (a ** -2 - 1.0)


def onebit(k, D, eps):
    """Effective noise of the one-bit randomizer."""
    C = 1.0 / np.tanh(eps / 2.0)
    return D ** 2 / 4.0 * (C ** 2 - (k + 1) / (3.0 * (k - 1)))


def piecewise(k, D, eps):
    """Effective noise of the piecewise mechanism."""
    e = np.exp(eps / 2.0); Et2 = (k + 1) / (3.0 * (k - 1))
    return D ** 2 / 4.0 * (Et2 / (e - 1.0) + (e + 3.0) / (3.0 * (e - 1.0) ** 2))


def hybrid(k, D, eps):
    """Effective noise of the hybrid mechanism."""
    tau = 1.839286755214161
    if eps <= np.log(tau):
        return onebit(k, D, eps)
    lam = 1.0 - np.exp(-eps / 2.0)
    return lam * piecewise(k, D, eps) + (1 - lam) * onebit(k, D, eps)


def _from_moments(g, m, m2):
    k = len(g); pi = np.full(k, 1.0 / k)
    EX, EY = pi @ g, pi @ m
    alpha = (pi @ ((g - EX) * (m - EY))) / (pi @ (g - EX) ** 2)
    V = pi @ (m2 - m ** 2)
    return alpha, V, V / alpha ** 2


def laplace(k, D, eps, lo, hi, n=400001):
    """Effective noise of clipped Laplace by numerical integration of the clipped density."""
    g = grid(k, lo, hi); b = D / eps
    y = np.linspace(lo, hi, n)
    m = np.empty(k); m2 = np.empty(k)
    for i, x in enumerate(g):
        f = np.exp(-np.abs(y - x) / b) / (2 * b)
        plo = 0.5 * np.exp(-(x - lo) / b)
        phi = 0.5 * np.exp(-(hi - x) / b)
        m[i] = _trapz(f * y, y) + plo * lo + phi * hi
        m2[i] = _trapz(f * y ** 2, y) + plo * lo ** 2 + phi * hi ** 2
    return _from_moments(g, m, m2)[2]


def geometric(k, D, eps, lo, hi):
    """Effective noise of the truncated geometric mechanism."""
    g = grid(k, lo, hi)
    a = np.exp(-eps / (k - 1.0)); c = (1 - a) / (1 + a)
    j = np.arange(k); m = np.empty(k); m2 = np.empty(k)
    for i in range(k):
        p = c * a ** np.abs(j - i)
        p[0] = c * a ** i / (1 - a)
        p[k - 1] = c * a ** (k - 1 - i) / (1 - a)
        p = p / p.sum()
        m[i] = p @ g
        m2[i] = p @ (g ** 2)
    return _from_moments(g, m, m2)[2]


def _alpha_c(g, m):
    k = len(g); pi = np.full(k, 1.0 / k)
    EX, EY = pi @ g, pi @ m
    alpha = (pi @ ((g - EX) * (m - EY))) / (pi @ (g - EX) ** 2)
    return alpha, EY - alpha * EX


def krr_atoms(k, D, eps, lo, hi):
    """Output alphabet of corrected k-RR, ``(g - c) / alpha``."""
    g = grid(k, lo, hi)
    a = (np.exp(eps) - 1.0) / (np.exp(eps) + k - 1.0)
    m = a * g + (1 - a) * g.mean()
    al, c = _alpha_c(g, m)
    return (g - c) / al


def geom_atoms(k, D, eps, lo, hi):
    """Output alphabet of the corrected truncated geometric mechanism."""
    g = grid(k, lo, hi)
    a = np.exp(-eps / (k - 1.0)); c0 = (1 - a) / (1 + a)
    j = np.arange(k); m = np.empty(k)
    for i in range(k):
        p = c0 * a ** np.abs(j - i)
        p[0] = c0 * a ** i / (1 - a); p[k - 1] = c0 * a ** (k - 1 - i) / (1 - a)
        p /= p.sum(); m[i] = p @ g
    al, c = _alpha_c(g, m)
    return (g - c) / al


def onebit_atoms(k, D, eps, lo, hi):
    """The two output atoms of the one-bit randomizer on the rating scale."""
    C = 1.0 / np.tanh(eps / 2.0)
    return np.array([lo + D / 2.0 * (-C + 1), lo + D / 2.0 * (C + 1)])


def _bisect(f, a, b, iters=200):
    fa = f(a)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if (fm > 0) == (fa > 0):
            a, fa = mid, fm
        else:
            b = mid
    return 0.5 * (a + b)


def crossovers(k, D=4.5, lo=1e-3, hi=40.0):
    """``(eps1, eps2)``: roots of ``rho_1 = 1`` (one-bit vs piecewise) and ``rho_2 = 1``."""
    e1 = _bisect(lambda e: np.log(onebit(k, D, e) / piecewise(k, D, e)), lo, hi)
    e2 = _bisect(lambda e: np.log(piecewise(k, D, e) / krr(k, D, e)), lo, hi)
    return e1, e2


def tilt_law(k, theta, gamma):
    """Exponential-tilt law ``pi(g_i) ~ exp(theta z_i + gamma (z_i - 1/2)^2)``, ``z_i = i/(k-1)``."""
    z = np.arange(k) / (k - 1.0)
    w = np.exp(theta * z + gamma * (z - 0.5) ** 2)
    return w / w.sum()


def crossover_under_law(grid_, lo, hi, pi, a="pm", b="krr", e_lo=0.05, e_hi=20.0):
    """Budget at which mechanisms ``a`` and ``b`` exchange under reference law ``pi``."""
    def f(e):
        return np.log(CH.channel(a, grid_, e, lo, hi, pi)[2] / CH.channel(b, grid_, e, lo, hi, pi)[2])
    return _bisect(f, e_lo, e_hi, iters=100)


def tilt_sweep(grid_, lo, hi, thetas, gammas, a="pm", b="krr"):
    """Range of the ``a``-vs-``b`` crossover over the tilt family."""
    vals = [crossover_under_law(grid_, lo, hi, tilt_law(len(grid_), th, ga), a, b)
            for th in thetas for ga in gammas]
    return float(min(vals)), float(max(vals))


def law_free(m1, m2, grid_, eps, lo, hi):
    """True if ``m1`` beats ``m2`` under every law on the grid by the interval condition.

    Effective noise under a law ``pi`` is the ``pi``-average of the per-level quantity
    ``v(x) = Var(Y | X = x) / alpha^2``, so it lies in ``[min v, max v]``. If
    ``max v_{m1} < min v_{m2}`` the two intervals are disjoint and ``m1`` has the smaller
    effective noise whatever the law (the law-free corollary).
    """
    s1 = CH.conditional_variance(m1, grid_, eps, lo, hi)
    s2 = CH.conditional_variance(m2, grid_, eps, lo, hi)
    return bool(s1.max() < s2.min())


def law_free_boundary(m1, m2, grid_, lo, hi, e_lo, e_hi):
    """Budget in ``[e_lo, e_hi]`` at which the law-free condition switches, by bisection."""
    def f(e):
        s1 = CH.conditional_variance(m1, grid_, e, lo, hi)
        s2 = CH.conditional_variance(m2, grid_, e, lo, hi)
        return np.log(s1.max() / s2.min())
    return _bisect(f, e_lo, e_hi, iters=100)


def law_free_threshold(m1, m2, grid_, lo, hi, e_lo=0.01, e_hi=20.0, n=4000):
    """Budgets on a fine log grid at which ``m1`` dominates ``m2`` under every law."""
    es = np.exp(np.linspace(np.log(e_lo), np.log(e_hi), n))
    return es, np.array([law_free(m1, m2, grid_, e, lo, hi) for e in es])
