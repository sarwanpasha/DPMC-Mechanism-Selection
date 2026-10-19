"""The exact optimum over the affine-mean class, and randomized response on bins.

Lower bound. After the public correction, a channel ``Q`` with ``E[Y | X] = c + alpha X``
becomes ``Y' = (Y - c) / alpha``, which is eps-DP by post-processing, unbiased
(``E[Y' | X] = X``) and has ``Var(Y' | X) = Var(Y | X) / alpha^2``. Its mean squared
error is therefore exactly ``sigma2_eff(Q)``, so the minimum mean squared error over all
unbiased eps-DP channels lower-bounds ``sigma2_eff`` over the whole affine-mean class.
On a fixed output alphabet that minimum is a linear program (Badanidiyuru et al.,
NeurIPS 2023): the objective and the row-sum, unbiasedness and privacy constraints are
all linear in the transition probabilities. Privacy is imposed through a lower envelope
``m_j <= P_ij <= e^eps m_j``, which needs ``2 k L`` inequalities instead of ``k(k-1)L``.

RR-on-bins. Ghazi et al. (ICLR 2023) show the optimal label-DP randomizer for a unimodal
loss is randomized response on bins: a monotone map from levels to ``b`` bins followed by
``b``-ary randomized response over bin representatives. For squared loss the
representatives are available in closed form given the bins, and on grids with ``k <= 10``
the bin boundaries are enumerated exactly.

Tables. ``optimal_tables.txt`` holds the transition matrices and output alphabets of both
channels on the two evaluation grids at the seven budgets, exactly as used in the
end-to-end runs. Each line is ``name rows,cols value value ...`` with names
``{lpopt,rrbins}_{z,P}_{ml,am}_{eps}``. ``experiments/make_optimal_tables.py`` regenerates
them from this module.
"""
import os
from itertools import combinations

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TAB = None
_GRID_TAG = {10: "ml", 5: "am"}


def opt_unbiased(g, eps, pi=None, L=161, span=None, extra=()):
    """Minimum MSE over unbiased eps-DP channels on a fixed output alphabet.

    The alphabet is ``L`` equally spaced points around the grid midpoint (half-width
    ``span``, by default enough to contain the one-bit atoms), together with the grid and
    any ``extra`` points. Returns ``(value, P, z)`` with ``P`` of shape ``(k, len(z))``;
    ``value`` is the optimal ``sum_i pi_i sum_j P_ij (z_j - g_i)^2``.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, lil_matrix

    k = len(g)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float)
    mid = 0.5 * (g[0] + g[-1]); D = g[-1] - g[0]
    if span is None:
        span = (1.0 / np.tanh(eps / 2.0) + 2.0) * D / 2.0
    z = np.unique(np.concatenate([np.linspace(mid - span, mid + span, L), g,
                                  np.asarray(extra, float)]))
    L = len(z)
    nv = k * L + L

    def pidx(i, j):
        return i * L + j

    def midx(j):
        return k * L + j

    cost = np.zeros(nv)
    for i in range(k):
        cost[pidx(i, 0):pidx(i, 0) + L] = pi[i] * (z - g[i]) ** 2

    Aeq, beq = lil_matrix((2 * k, nv)), np.zeros(2 * k)
    for i in range(k):
        Aeq[i, pidx(i, 0):pidx(i, 0) + L] = 1.0
        beq[i] = 1.0
        Aeq[k + i, pidx(i, 0):pidx(i, 0) + L] = z
        beq[k + i] = g[i]

    Aub = lil_matrix((2 * k * L, nv)); bub = np.zeros(2 * k * L)
    r = 0
    ee = np.exp(eps)
    for i in range(k):
        for j in range(L):
            Aub[r, pidx(i, j)] = -1.0; Aub[r, midx(j)] = 1.0; r += 1
            Aub[r, pidx(i, j)] = 1.0; Aub[r, midx(j)] = -ee; r += 1
    res = linprog(cost, A_ub=csr_matrix(Aub), b_ub=bub, A_eq=csr_matrix(Aeq), b_eq=beq,
                  bounds=[(0, None)] * nv, method="highs")
    if not res.success:
        return np.nan, None, z
    P = res.x[:k * L].reshape(k, L)
    return res.fun, P, z


def rr_bins(g, eps, pi=None, bmax=None):
    """Best randomized response on bins for squared loss.

    Enumerates every monotone partition of the ``k`` levels into ``b`` bins for
    ``b = 2, ..., bmax`` (default ``k``), sets the representatives to their squared-loss
    optimum, and returns ``(z, P, b, mse)`` for the best partition.
    """
    k = len(g)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float)
    best = None
    for b in range(2, (bmax or k) + 1):
        p = np.exp(eps) / (np.exp(eps) + b - 1.0)
        q = 1.0 / (np.exp(eps) + b - 1.0)
        for cuts in combinations(range(1, k), b - 1):
            edges = (0,) + cuts + (k,)
            bin_of = np.zeros(k, dtype=int)
            for t in range(b):
                bin_of[edges[t]:edges[t + 1]] = t
            P = np.full((k, b), q)
            P[np.arange(k), bin_of] = p
            A = (pi[:, None] * P).sum(axis=0)
            Bv = (pi[:, None] * P * g[:, None]).sum(axis=0)
            z = Bv / A
            val = float((pi[:, None] * P * (z[None, :] - g[:, None]) ** 2).sum())
            if best is None or val < best[0]:
                best = (val, z.copy(), P.copy(), b, edges)
    val, z, P, b, edges = best
    return z, P, b, val


def _load():
    global _TAB
    if _TAB is not None:
        return _TAB
    tab = {}
    with open(os.path.join(_HERE, "optimal_tables.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, shape, vals = line.split(" ", 2)
            shp = tuple(int(x) for x in shape.split(","))
            tab[name] = np.array(vals.split(), dtype=float).reshape(shp)
    _TAB = tab
    return tab


def table(mech, grid, eps):
    """Output alphabet ``z`` and transition matrix ``P`` of ``mech`` on ``grid`` at ``eps``."""
    k = len(grid)
    if k not in _GRID_TAG:
        raise ValueError("tables are provided for the 10-level and 5-level grids; "
                         "run experiments/make_optimal_tables.py for other grids")
    t = _load()
    key = "%s_%g" % (_GRID_TAG[k], eps)
    return t["%s_z_%s" % (mech, key)], t["%s_P_%s" % (mech, key)]


def table_stats(mech, grid, eps, pi=None):
    """Channel statistics ``(alpha, V, sigma2_eff, c)`` of a tabulated channel."""
    from .channels import _stats
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float) / np.sum(pi)
    z, P = table(mech, grid, eps)
    Ey = P @ z
    return _stats(Ey, (P @ z ** 2) - Ey ** 2, grid, pi)


def table_sample(mech, vals, grid, eps, rng):
    """Privatize ratings with a tabulated channel."""
    z, P = table(mech, grid, eps)
    idx = np.abs(vals[:, None] - grid[None, :]).argmin(axis=1)
    C = np.cumsum(P[idx], axis=1)
    u = rng.random(len(idx))[:, None]
    j = (u > C).sum(axis=1)
    return z[np.clip(j, 0, len(z) - 1)]


def privacy_loss(P):
    """Realized privacy loss ``max_j log(max_i P_ij / min_i P_ij)`` of a transition matrix."""
    return float(np.log(P.max(axis=0) / np.maximum(P.min(axis=0), 1e-300)).max())


def affine_residual(grid, z, P, pi=None):
    """Residual of the affine correction: ``(alpha, c, sup|r|, E[r^2]/alpha^2)``."""
    k = len(grid)
    pi = np.full(k, 1.0 / k) if pi is None else np.asarray(pi, float)
    m = P @ z
    EX, EY = pi @ grid, pi @ m
    alpha = (pi @ ((grid - EX) * (m - EY))) / (pi @ (grid - EX) ** 2)
    c = EY - alpha * EX
    r = m - (c + alpha * grid)
    return float(alpha), float(c), float(np.abs(r).max()), float(pi @ (r ** 2)) / alpha ** 2


def selftest():
    """Check that every tabulated channel is a stochastic matrix within its privacy budget."""
    for k, lo, hi in ((10, 0.5, 5.0), (5, 1.0, 5.0)):
        g = np.linspace(lo, hi, k)
        for eps in (0.1, 0.25, 0.5, 1, 2, 4, 8):
            for m in ("lpopt", "rrbins"):
                z, P = table(m, g, eps)
                assert np.allclose(P.sum(axis=1), 1, atol=1e-9), (m, k, eps, "rows")
                loss = privacy_loss(P)
                assert loss <= eps + 1e-6, (m, k, eps, "privacy", loss)
                a, V, s2, c = table_stats(m, g, eps)
                print("%-6s k=%-3d eps=%-5g loss=%.6f alpha=%.5f sigma2_eff=%.6g"
                      % (m, k, eps, loss, a, s2))
    print("optimal tables: all rows stochastic and within budget")


if __name__ == "__main__":
    selftest()
