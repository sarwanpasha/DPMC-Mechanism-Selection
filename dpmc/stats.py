"""Paired t-test and rank correlation without a scipy dependency.

The two-sided p-value of Student's t is the regularized incomplete beta function
``I_{nu/(nu+t^2)}(nu/2, 1/2)``, evaluated by Lentz's continued fraction.
"""
import math

import numpy as np


def _betacf(a, b, x, itmax=300, eps=3e-16, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def two_sided_p(t, df):
    """Two-sided p-value of Student's t with ``df`` degrees of freedom."""
    if df <= 0 or not math.isfinite(t):
        return float('nan')
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def t_critical(df, level=0.95):
    """Two-sided critical value of Student's t, by bisection on ``two_sided_p``."""
    target = 1.0 - level
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if two_sided_p(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def paired_test(a, b, level=0.95):
    """Paired comparison of two equal-length samples.

    Returns ``(mean_diff, half_width, p)``: the mean of ``a - b``, the half-width of its
    two-sided confidence interval at ``level``, and the paired t-test p-value. Identical
    samples give ``(0, 0, nan)``.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n < 2:
        return float(d.mean()) if n else float('nan'), float('nan'), float('nan')
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return float(d.mean()), 0.0, float('nan')
    se = sd / math.sqrt(n)
    t = float(d.mean()) / se
    return float(d.mean()), t_critical(n - 1, level) * se, two_sided_p(t, n - 1)


def spearman(a, b):
    """Spearman rank correlation (ties broken by order of appearance)."""
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra.dot(rb) / math.sqrt(ra.dot(ra) * rb.dot(rb)))


def selftest():
    """Check p-values against standard t-table critical values."""
    cases = [(2.776, 4, 0.050), (2.132, 4, 0.100), (4.604, 4, 0.010),
             (2.571, 5, 0.050), (1.960, 10 ** 9, 0.050), (3.182, 3, 0.050)]
    ok = True
    for t, df, want in cases:
        got = two_sided_p(t, df)
        good = abs(got - want) < 5e-4
        ok &= good
        print('t=%-7g df=%-10g p=%.5f want %.3f %s' % (t, df, got, want, 'OK' if good else 'FAIL'))
    print('t-test selftest:', 'PASS' if ok else 'FAIL')
    return ok


if __name__ == '__main__':
    selftest()
