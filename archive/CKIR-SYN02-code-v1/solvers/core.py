"""Protocol v0.3 operators. All solvers are CPU float64, labels not used in QP.

This is the proposed residual-probability model, NOT a faithful ProKeR baseline.
"""
from dataclasses import dataclass
from itertools import product
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class NumericalFailure(RuntimeError):
    pass


def maxabs(x):
    return float(np.max(np.abs(x), initial=0.0))


def kernel(x, y, beta):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if beta <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("illegal kernel input")
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError("support features are not normalized")
    if not np.allclose(np.linalg.norm(y, axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError("query features are not normalized")
    return np.exp(-float(beta) * (1.0 - np.clip(x @ y.T, -1.0, 1.0)))


def regularized_kernel(features, beta):
    raw = kernel(features, features, beta)
    raw = (raw + raw.T) / 2
    for eta in (0., 1e-10, 1e-8, 1e-6):
        adjusted = raw + eta * np.eye(len(raw))
        try:
            cho_factor(adjusted, lower=True, check_finite=True)
            return adjusted, eta
        except np.linalg.LinAlgError:
            continue
    raise NumericalFailure("kernel Cholesky failed at all predeclared jitters")


def objective(w, a, k):
    return float(w @ a @ w - 2 * k @ w)


def projected_gradient(w, a, k, tau):
    grad = 2 * (a @ w - k)
    return maxabs(w - np.clip(w - grad, -tau, tau))


@dataclass
class QPResult:
    w: np.ndarray
    accepted: bool
    seconds: float
    iterations: int
    evaluations: int
    attempts: int
    pg_inf: float
    pg_tolerance: float
    box_violation: float
    objective: float
    clip_objective: float
    objective_ok: bool
    message: str

    def summary(self):
        return {k: v for k, v in vars(self).items() if k != "w"}


def solve_box(a, k, tau, w0=None):
    a = np.asarray(a, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    if tau < 0 or not np.isfinite(a).all() or not np.isfinite(k).all():
        raise ValueError("illegal QP input")
    if a.shape != (len(k), len(k)) or not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("A must be symmetric square")
    if w0 is None:
        w0 = cho_solve(cho_factor(a, lower=True), k)
    initial = np.clip(np.asarray(w0, dtype=np.float64), -tau, tau)
    qclip = objective(initial, a, k)
    tolerance = 1e-7 * (1 + maxabs(k))
    start = time.perf_counter()
    iterations = evaluations = 0
    for attempt in (1, 2):
        multiplier = attempt
        result = minimize(
            lambda w: (objective(w, a, k), 2 * (a @ w - k)),
            initial.copy(), jac=True, method="L-BFGS-B",
            bounds=[(-tau, tau)] * len(k),
            options={"maxiter": 2000 * multiplier, "maxfun": 10000 * multiplier,
                     "ftol": 1e-12, "gtol": 1e-10},
        )
        w = np.asarray(result.x, dtype=np.float64)
        pg = projected_gradient(w, a, k, tau)
        violation = max(0., maxabs(w) - tau)
        q = objective(w, a, k)
        objective_ok = q <= qclip + 1e-8 * (1 + abs(qclip))
        iterations += int(result.get("nit", 0))
        evaluations += int(result.nfev)
        accepted = bool(np.isfinite(w).all() and pg <= tolerance
                        and violation <= 1e-10 and objective_ok)
        if accepted:
            break
    return QPResult(w, accepted, time.perf_counter() - start, iterations,
                    evaluations, attempt, pg, tolerance, violation, q,
                    qclip, bool(objective_ok), str(result.message))


def enumerate_box(a, k, tau):
    """Independent active-set oracle: enumerate all 3**n status vectors, n<=8."""
    a, k = np.asarray(a, dtype=np.float64), np.asarray(k, dtype=np.float64)
    if len(k) > 8:
        raise ValueError("oracle is restricted to n<=8")
    best, best_q = None, np.inf
    for pattern in product((-1, 0, 1), repeat=len(k)):
        status = np.asarray(pattern)
        fixed, free = status != 0, status == 0
        w = tau * status.astype(np.float64)
        if np.any(free):
            w[free] = np.linalg.solve(a[np.ix_(free, free)],
                                    k[free] - a[np.ix_(free, fixed)] @ w[fixed])
        if maxabs(w) > tau + 1e-9:
            continue
        grad = 2 * (a @ w - k)
        if (np.any(grad[status == -1] < -1e-8)
                or np.any(grad[status == 1] > 1e-8)
                or maxabs(grad[free]) > 1e-8):
            continue
        q = objective(w, a, k)
        if q < best_q:
            best, best_q = w.copy(), q
    if best is None:
        raise NumericalFailure("independent oracle found no KKT solution")
    return best


def huber_value_grad(g_flat, k, residual, lam, delta):
    n = len(k)
    g = g_flat.reshape(residual.shape)
    error = k @ g - residual
    abs_error = np.abs(error)
    loss = np.where(abs_error <= delta, error ** 2 / 2,
                    delta * abs_error - delta ** 2 / 2).sum()
    kg = k @ g
    value = (loss + lam / 2 * np.sum(g * kg)) / n
    grad = (k.T @ np.clip(error, -delta, delta) + lam * kg) / n
    return float(value), grad.ravel()


def fit_huber(k, residual, lam, delta):
    k, residual = np.asarray(k, dtype=np.float64), np.asarray(residual, dtype=np.float64)
    ridge = cho_solve(cho_factor(k + lam * np.eye(len(k)), lower=True), residual)
    tolerance = 1e-7 * max(1., maxabs(k @ residual) / len(k))
    start = time.perf_counter()
    total_iterations = total_evaluations = 0
    for attempt in (1, 2):
        result = minimize(huber_value_grad, ridge.ravel().copy(),
                          args=(k, residual, lam, delta), jac=True, method="L-BFGS-B",
                          options={"maxiter": 2000 * attempt, "maxfun": 10000 * attempt,
                                   "ftol": 1e-12, "gtol": 1e-10})
        value, grad = huber_value_grad(result.x, k, residual, lam, delta)
        norm = maxabs(grad)
        total_iterations += int(result.nit)
        total_evaluations += int(result.nfev)
        accepted = bool(np.isfinite(result.x).all() and norm <= tolerance)
        if accepted:
            break
    return result.x.reshape(residual.shape), {
        "accepted": accepted, "seconds": time.perf_counter() - start,
        "iterations": total_iterations, "evaluations": total_evaluations,
        "attempts": attempt, "grad_inf": norm, "grad_tolerance": tolerance,
        "objective": value, "message": str(result.message),
    }


def residuals(observed_labels, p0):
    return np.eye(p0.shape[1])[observed_labels] - p0


def class_balanced_metrics(labels, zero_pred, pred, classes):
    labels, zero_pred, pred = map(np.asarray, (labels, zero_pred, pred))
    counts = np.bincount(labels, minlength=classes)
    if len(counts) != classes or np.any(counts == 0):
        raise ValueError("missing or illegal query class")
    weights = 1. / (classes * counts[labels])
    correct0, correct = zero_pred == labels, pred == labels
    a0, accuracy = float(weights @ correct0), float(weights @ correct)
    errors, successes = float(weights @ ~correct0), float(weights @ correct0)
    benefit = float(weights @ (~correct0 & correct)) / errors if errors else None
    harm = float(weights @ (correct0 & ~correct)) / successes if successes else None
    if benefit is not None and harm is not None:
        identity = a0 * (1 - harm) + (1 - a0) * benefit
        if abs(identity - accuracy) > 1e-12:
            raise NumericalFailure("metric identity failed")
    return {"a0": a0, "accuracy": accuracy, "B": benefit, "H": harm,
            "zero_error_weight": errors, "zero_correct_weight": successes}


def paired_block_bootstrap(differences, replicates=10000, seed=424242):
    """Input = already-noise-averaged support blocks x endpoints; common indices."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim == 1:
        differences = differences[:, None]
    if not np.isfinite(differences).all():
        raise ValueError("undefined endpoint blocks inference")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(differences), (replicates, len(differences)))
    means = differences[indices].mean(axis=1)
    intervals = np.quantile(means, [.025, .975], axis=0, method="linear")
    degenerate = np.ptp(differences, axis=0) == 0
    return {"point": differences.mean(axis=0).tolist(),
            "lower": intervals[0].tolist(), "upper": intervals[1].tolist(),
            "inconclusive_degenerate": degenerate.tolist(),
            "independent_blocks": len(differences), "replicates": replicates,
            "seed": seed}
