"""Bias-aware matrix factorization fitted by alternating least squares.

Model: ``R_ui ~ mu + b_u + c_i + <p_u, q_i>`` with ridge penalty ``reg`` on factors and
biases. Each half-step solves, for every row (or column) independently, the regularized
least-squares problem whose design is the other side's factors with a column of ones
appended; this is the subproblem analysed by the least-squares risk identity.
"""
import numpy as np


def _group(index, n_groups):
    """Observation indices grouped by row (or column) id."""
    order = np.argsort(index, kind="stable")
    bounds = np.searchsorted(index[order], np.arange(n_groups + 1))
    return [order[bounds[g]:bounds[g + 1]] for g in range(n_groups)]


def _side_update(groups, other_F, other_b, mu, vals, partner, rank, reg):
    """Joint ridge update of one side's factors and biases with the other side fixed."""
    F = np.zeros((len(groups), rank))
    b = np.zeros(len(groups))
    ridge = reg * np.eye(rank + 1)
    for s, sel in enumerate(groups):
        if sel.size == 0:
            continue
        X = np.hstack([other_F[partner[sel]], np.ones((sel.size, 1))])
        y = vals[sel] - mu - other_b[partner[sel]]
        sol = np.linalg.solve(X.T @ X + ridge, X.T @ y)
        F[s], b[s] = sol[:rank], sol[rank]
    return F, b


def fit_biased_mf(rows, cols, vals, shape, rank=32, reg=5.0, n_iter=15,
                  clip=None, val=None, verbose=False):
    """Fit the biased factorization by ALS.

    Factors are initialized at ``0.01 * N(0, 1)`` from a fixed generator, so a fit is a
    deterministic function of the data. ``val = (rows, cols, vals)`` records validation
    RMSE after each iteration. Returns a dict with ``mu, bu, bi, P, Q``, a
    ``predict(rows, cols)`` callable that clips to ``clip``, and the validation history.
    """
    m, n = shape
    rows = np.asarray(rows); cols = np.asarray(cols)
    vals = np.asarray(vals, dtype=float)
    mu = float(vals.mean())
    rng = np.random.default_rng(0)
    P = 0.01 * rng.standard_normal((m, rank))
    Q = 0.01 * rng.standard_normal((n, rank))
    bu = np.zeros(m); bi = np.zeros(n)
    ugroups = _group(rows, m)
    igroups = _group(cols, n)
    lo, hi = (clip if clip else (-np.inf, np.inf))

    def predict(r, c):
        out = mu + bu[r] + bi[c] + np.einsum('ij,ij->i', P[r], Q[c])
        return np.clip(out, lo, hi)

    history = []
    for it in range(n_iter):
        P, bu = _side_update(ugroups, Q, bi, mu, vals, cols, rank, reg)
        Q, bi = _side_update(igroups, P, bu, mu, vals, rows, rank, reg)
        if val is not None:
            vr, vc, vv = val
            e = float(np.sqrt(np.mean((predict(vr, vc) - vv) ** 2)))
            history.append(e)
            if verbose:
                print(f"  [als] iter={it+1:2d} val_rmse={e:.4f}", flush=True)
    return dict(mu=mu, bu=bu, bi=bi, P=P, Q=Q, predict=predict,
                history=history, rank=rank, reg=reg)


def fit_bias_only(rows, cols, vals, shape, reg=10.0, n_iter=15, clip=None):
    """Additive model ``mu + b_u + c_i`` without an interaction term."""
    m, n = shape
    vals = np.asarray(vals, dtype=float)
    mu = float(vals.mean())
    bu = np.zeros(m); bi = np.zeros(n)
    ucnt = np.bincount(rows, minlength=m)
    icnt = np.bincount(cols, minlength=n)
    lo, hi = (clip if clip else (-np.inf, np.inf))
    for _ in range(n_iter):
        bu = np.bincount(rows, weights=vals - mu - bi[cols], minlength=m) / (reg + ucnt)
        bi = np.bincount(cols, weights=vals - mu - bu[rows], minlength=n) / (reg + icnt)

    def predict(r, c):
        return np.clip(mu + bu[r] + bi[c], lo, hi)

    return dict(mu=mu, bu=bu, bi=bi, predict=predict)


def user_half_step(item_F, item_b, mu, cols_u, x_u, reg):
    """Ridge solution of one user's half-step and its design, given item factors.

    Returns ``(theta, A, y)`` where ``A = [q_j, 1]`` over the user's items, ``y`` is the
    corrected release minus ``mu + c_j``, and ``theta = (A^T A + reg I)^{-1} A^T y`` holds
    ``(p_u, b_u)``.
    """
    rank = item_F.shape[1]
    A = np.hstack([item_F[cols_u], np.ones((cols_u.size, 1))])
    y = x_u - mu - item_b[cols_u]
    theta = np.linalg.solve(A.T @ A + reg * np.eye(rank + 1), A.T @ y)
    return theta, A, y
