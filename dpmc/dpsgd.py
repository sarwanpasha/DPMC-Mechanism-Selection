"""Gradient-perturbation baselines for the biased factorization.

Full-batch gradient descent with per-example clipping of the joint gradient of
``(p_u, q_i, b_u, c_i)`` to norm ``C``. One rating enters exactly one gradient sum per
epoch, so ``T`` epochs are ``T`` adaptive applications of the noise mechanism.

* ``gauss``: Gaussian noise of scale ``sigma C``, accounted by Renyi DP,
  ``RDP(a) = T a / (2 sigma^2)``, converted to ``(eps, delta)``-DP.
* ``laplace_pure``: Laplace noise calibrated to the l1 sensitivity of the summed clipped
  gradient. A rating touches ``2(d + 1)`` coordinates, so the l1 norm of a clipped
  per-example gradient is at most ``C sqrt(2(d + 1))``, and the scale for pure
  ``eps``-DP over ``T`` epochs is ``C sqrt(2(d + 1)) T / eps``.

The global offset ``mu`` is the public midpoint of the rating range, so no
data-dependent centring is needed, and the weight decay acts on already-private
parameters, so it is post-processing.
"""
import numpy as np

ORDERS = ([1.0 + x / 10.0 for x in range(1, 100)] + list(range(2, 512))
          + [1024, 4096, 16384, 65536, 262144, 1048576])


def rdp_gaussian_eps(sigma, steps, delta, orders=None):
    """``(eps, delta)`` certified by composing ``steps`` Gaussian mechanisms of multiplier ``sigma``."""
    if orders is None:
        orders = ORDERS
    best = float("inf")
    for a in orders:
        if a <= 1:
            continue
        rdp = steps * a / (2.0 * sigma ** 2)
        best = min(best, rdp + np.log(1.0 / delta) / (a - 1.0))
    return float(best)


def sigma_for_target_eps(target_eps, steps, delta, lo=0.1, hi=1e4, iters=80):
    """Smallest noise multiplier whose certified epsilon is at most ``target_eps``."""
    for _ in range(40):
        if rdp_gaussian_eps(hi, steps, delta) <= target_eps:
            break
        hi *= 2.0
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        if rdp_gaussian_eps(mid, steps, delta) <= target_eps:
            b = mid
        else:
            a = mid
    return b


def laplace_scale_pure(clipC, rank, epochs, eps):
    """Laplace scale giving pure ``eps``-DP over ``epochs`` full-batch steps."""
    return clipC * np.sqrt(2.0 * (rank + 1)) * epochs / float(eps)


def dpsgd_mf(rows, cols, vals, shape, rank, reg, lr, epochs, clipC, sigma, rng, clip,
             noise_kind='gauss', b_lap=None):
    """Train the biased factorization privately; returns a ``predict(rows, cols)`` callable."""
    m, n = shape
    mu = 0.5 * (clip[0] + clip[1])
    P = 0.01 * rng.standard_normal((m, rank))
    Q = 0.01 * rng.standard_normal((n, rank))
    bu = np.zeros(m)
    bi = np.zeros(n)
    for _ in range(epochs):
        Pi, Qj = P[rows], Q[cols]
        e = vals - (mu + bu[rows] + bi[cols] + np.einsum("ij,ij->i", Pi, Qj))
        nrm = np.abs(e) * np.sqrt((Qj ** 2).sum(1) + (Pi ** 2).sum(1) + 2.0)
        scale = np.minimum(1.0, clipC / np.maximum(nrm, 1e-12))
        es = e * scale
        gP = np.zeros_like(P); gQ = np.zeros_like(Q)
        for d in range(rank):
            gP[:, d] = -np.bincount(rows, weights=es * Qj[:, d], minlength=m)
            gQ[:, d] = -np.bincount(cols, weights=es * Pi[:, d], minlength=n)
        gbu = -np.bincount(rows, weights=es, minlength=m)
        gbi = -np.bincount(cols, weights=es, minlength=n)
        if noise_kind == 'gauss':
            s = sigma * clipC
            gP += rng.normal(0.0, s, size=gP.shape)
            gQ += rng.normal(0.0, s, size=gQ.shape)
            gbu += rng.normal(0.0, s, size=gbu.shape)
            gbi += rng.normal(0.0, s, size=gbi.shape)
        else:
            gP += rng.laplace(0.0, b_lap, size=gP.shape)
            gQ += rng.laplace(0.0, b_lap, size=gQ.shape)
            gbu += rng.laplace(0.0, b_lap, size=gbu.shape)
            gbi += rng.laplace(0.0, b_lap, size=gbi.shape)
        decay = 1.0 - lr * reg
        P = decay * P - lr * gP
        Q = decay * Q - lr * gQ
        bu = decay * bu - lr * gbu
        bi = decay * bi - lr * gbi

    def predict(r, c):
        out = mu + bu[r] + bi[c] + np.einsum("ij,ij->i", P[r], Q[c])
        return np.clip(out, clip[0], clip[1])
    return predict
