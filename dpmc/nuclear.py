"""Nuclear-norm reconstruction by ADMM, used as the reconstruction ablation.

Solves ``min_X f_Omega(X) + lam ||X||_*`` through the consensus split
``min_{X,Z} f(Z) + lam ||X||_*`` subject to ``X = Z`` with a scaled dual. ``f`` is a
weighted l1 (``fidelity="l1"``) or squared (``fidelity="l2"``) data term on the observed
entries and zero elsewhere.
"""
import numpy as np


def laplace_scale(sensitivity, eps):
    """Scale of the Laplace mechanism."""
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return sensitivity / eps


def svd_soft_threshold(A, tau):
    """Proximal operator of ``tau ||.||_*``: singular-value soft-thresholding."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_th = np.maximum(s - tau, 0.0)
    return (U * s_th) @ Vt, s_th


def dense_from_obs(rows, cols, vals, shape, fill=0.0):
    """Dense matrix holding the observed values and ``fill`` elsewhere."""
    M = np.full(shape, fill, dtype=float)
    M[rows, cols] = vals
    return M


def admm_nuclear(rows, cols, mtilde, shape, lam, fidelity="l1", weights=None,
                 rho=1.0, max_iter=300, tol=1e-5, verbose=False):
    """Return ``(X, converged, iterations, numerical_rank)``."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    mtilde = np.asarray(mtilde, dtype=float)
    if weights is None:
        weights = np.ones(mtilde.shape[0])
    weights = np.asarray(weights, dtype=float)
    obs = (rows, cols)
    fill = float(np.mean(mtilde)) if mtilde.size else 0.0
    X = dense_from_obs(rows, cols, mtilde, shape, fill=fill)
    Z = X.copy()
    U = np.zeros(shape)
    converged = False
    last_sv = np.zeros(min(shape))
    for it in range(max_iter):
        X, last_sv = svd_soft_threshold(Z - U, lam / rho)
        Z = X + U
        Zprev = Z.copy()
        xu_obs = (X + U)[obs]
        if fidelity == "l1":
            thr = weights / rho
            d = xu_obs - mtilde
            Z[obs] = mtilde + np.sign(d) * np.maximum(np.abs(d) - thr, 0.0)
        else:
            Z[obs] = (2.0 * weights * mtilde + rho * xu_obs) / (2.0 * weights + rho)
        U = U + (X - Z)
        r_primal = np.linalg.norm(X - Z)
        s_dual = rho * np.linalg.norm(Z - Zprev)
        if verbose and (it % 25 == 0):
            print(f"    [admm] it={it} rp={r_primal:.2e} sd={s_dual:.2e}", flush=True)
        if (r_primal < tol * max(1.0, np.linalg.norm(X)) and
                s_dual < tol * max(1.0, np.linalg.norm(U))):
            converged = True
            break
    rank = int(np.sum(last_sv > 1e-3 * (last_sv.max() if last_sv.size else 1.0)))
    return X, converged, it + 1, rank


def default_lambda(delta, eps2, n_obs, m, n):
    """Noise-calibrated weight ``sqrt(2) Delta / eps2 * sqrt((m + n) / |Omega|)``."""
    varsigma = np.sqrt(2.0) * delta / eps2
    return float(varsigma * np.sqrt((m + n) / max(n_obs, 1)))
