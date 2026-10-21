"""Honest confidence: temperature scaling, ECE, risk-controlled auto-accept threshold, conformal sets.

Guarantees below hold only if future inputs are exchangeable with the calibration set.
Distribution shift breaks them; that is why the engine keeps auditing.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, t):
    p = softmax(logits / t)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits, y):
    r = minimize_scalar(lambda lt: nll(logits, y, np.exp(lt)), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


def cp_upper(errors, n, delta):
    """One-sided Clopper-Pearson upper bound on an error rate."""
    return 1.0 if n == 0 else float(beta.ppf(1 - delta, errors + 1, n - errors)) if errors < n else 1.0


def risk_threshold(conf, correct, alpha=0.05, delta=0.05, grid=200):
    """Learn-then-Test style fixed-sequence testing (strict -> loose thresholds).

    Returns the loosest confidence threshold whose accepted-set error rate is bounded
    by alpha with probability >= 1-delta. Stops at the first failure, which keeps the
    family-wise error control valid without a multiplicity penalty.
    """
    # Smallest accepted-set size at which even zero errors could pass; start the sequence at 2x that
    # (the ordering depends on confidences only, never on labels, so it is fixed before testing).
    n_start = 2 * int(np.ceil(np.log(delta) / np.log(1 - alpha)))
    taus = np.quantile(conf, np.linspace(1, 0, grid))
    best = None
    for tau in taus:
        m = conf >= tau
        n, err = int(m.sum()), int((~correct[m]).sum())
        if n < n_start:
            continue
        if cp_upper(err, n, delta) <= alpha:
            best = float(tau)
        else:
            break
    return best  # None => never auto-accept at this risk level


def conformal_qhat(probs, y, alpha=0.1):
    """Split-conformal (LAC). Set {k: p_k >= 1-qhat} covers the truth w.p. >= 1-alpha."""
    s = 1 - probs[np.arange(len(y)), y]
    n = len(s)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(np.sort(s)[k - 1])
