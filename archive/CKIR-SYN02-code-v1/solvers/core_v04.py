"""Numerical-only v0.4 revision. Original objectives and acceptance are unchanged."""
from dataclasses import dataclass, field
import time

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from core import maxabs, objective, projected_gradient, huber_value_grad


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
    extra: dict = field(default_factory=dict)

    def summary(self):
        return {k: v for k, v in vars(self).items() if k != "w"}


class BoxWorkspace:
    """Reusable OSQP factorization, shared geometry only. Reset primal/dual per query."""
    def __init__(self, a, tau):
        before = time.perf_counter()
        self.a = np.asarray(a, dtype=np.float64)
        self.tau = float(tau)
        if tau < 0 or not np.isfinite(tau) or not np.isfinite(self.a).all():
            raise ValueError("illegal QP input")
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1] or not np.allclose(self.a, self.a.T, atol=1e-12):
            raise ValueError("A must be symmetric square")
        self.n = len(self.a)
        self.solver = None
        self.factor = cho_factor(self.a, lower=True)
        self.setup_seconds = time.perf_counter() - before

    def _new_solver(self, k, maxiter):
        solver = osqp.OSQP(algebra="builtin")
        solver.setup(P=sparse.csc_matrix(np.triu(2 * self.a)), q=-2 * k,
                     A=sparse.eye(self.n, format="csc"),
                     l=np.full(self.n, -self.tau), u=np.full(self.n, self.tau),
                     verbose=False, eps_abs=1e-10, eps_rel=1e-10,
                     max_iter=maxiter, adaptive_rho=True, adaptive_rho_interval=50,
                     rho=.1, scaling=10, polishing=True, polish_refine_iter=10,
                     delta=1e-9, warm_starting=True, time_limit=30.)
        return solver

    def solve(self, k, w0=None):
        start = time.perf_counter()
        k = np.asarray(k, dtype=np.float64)
        if k.shape != (self.n,) or not np.isfinite(k).all():
            raise ValueError("illegal cross-kernel vector")
        if w0 is None:
            w0 = cho_solve(self.factor, k)
        initial = np.clip(np.asarray(w0, dtype=np.float64), -self.tau, self.tau)
        tolerance = 1e-7 * (1 + maxabs(k))
        qclip = objective(initial, self.a, k)
        initial_pg = projected_gradient(initial, self.a, k, self.tau)
        if initial_pg <= tolerance:
            return QPResult(initial, True, time.perf_counter() - start, 0, 0, 0,
                            initial_pg, tolerance, 0., qclip, qclip, True,
                            "initial point already satisfies locked KKT criterion",
                            {"engine": "OSQP-v04", "initial_shortcut": True})
        iterations = 0
        attempt_records = []
        for attempt in (1, 2):
            if self.solver is None or attempt == 2:
                solver = self._new_solver(k, 20000 * attempt)
                if attempt == 1:
                    self.solver = solver
            else:
                solver = self.solver
                solver.update(q=-2 * k)
                solver.update_settings(max_iter=20000)
            solver.warm_start(x=initial.copy(), y=np.zeros(self.n))
            result = solver.solve(raise_error=False)
            iterations += int(result.info.iter)
            finite = result.x is not None and np.isfinite(result.x).all()
            raw = np.asarray(result.x, dtype=np.float64) if finite else np.full(self.n, np.nan)
            raw_violation = max(0., maxabs(raw) - self.tau) if finite else None
            # Roundoff-only projection is audited; a badly infeasible solve cannot pass.
            w = np.clip(raw, -self.tau, self.tau) if finite else initial.copy()
            pg = projected_gradient(w, self.a, k, self.tau)
            q = objective(w, self.a, k)
            violation = max(0., maxabs(w) - self.tau)
            objective_ok = q <= qclip + 1e-8 * (1 + abs(qclip))
            accepted = bool(finite and raw_violation <= 1e-8 and pg <= tolerance
                            and violation <= 1e-10 and objective_ok)
            attempt_records.append({"attempt": attempt, "status": str(result.info.status),
                "status_polish": int(result.info.status_polish),
                "raw_box_violation": raw_violation,
                "projection_displacement": maxabs(w - raw) if finite else None,
                "iterations": int(result.info.iter), "pg_inf": pg,
                "accepted": accepted})
            if accepted:
                break
        return QPResult(w, accepted, time.perf_counter() - start, iterations,
                        iterations, attempt, pg, tolerance, violation, q, qclip,
                        bool(objective_ok), str(result.info.status),
                        {"engine": "OSQP-v04", "initial_shortcut": False,
                         "attempts_detail": attempt_records})


def solve_box(a, k, tau, w0=None):
    start = time.perf_counter()
    workspace = BoxWorkspace(a, tau)
    result = workspace.solve(k, w0)
    result.seconds = time.perf_counter() - start
    return result


def whitened_huber_value_grad(z_flat, lower, residual, lam, delta):
    z = z_flat.reshape(residual.shape)
    error = lower @ z - residual
    abs_error = np.abs(error)
    loss = np.where(abs_error <= delta, error ** 2 / 2,
                    delta * abs_error - delta ** 2 / 2).sum()
    value = loss + lam / 2 * np.sum(z * z)
    grad = lower.T @ np.clip(error, -delta, delta) + lam * z
    return float(value), grad.ravel()


def fit_huber(k, residual, lam, delta):
    start = time.perf_counter()
    k, residual = np.asarray(k, dtype=np.float64), np.asarray(residual, dtype=np.float64)
    if lam <= 0 or delta <= 0 or not np.isfinite(k).all() or not np.isfinite(residual).all():
        raise ValueError("illegal Huber input")
    n = len(k)
    lower = cholesky(k, lower=True)
    ridge = cho_solve(cho_factor(k + lam * np.eye(n), lower=True), residual)
    tolerance = 1e-7 * max(1., maxabs(k @ residual) / n)
    original_value, original_grad = huber_value_grad(ridge.ravel(), k, residual, lam, delta)
    norm = maxabs(original_grad)
    if norm <= tolerance:
        return ridge, {"accepted": True, "seconds": time.perf_counter() - start,
            "iterations": 0, "evaluations": 1, "attempts": 0, "grad_inf": norm,
            "grad_tolerance": tolerance, "objective": original_value,
            "message": "ridge already satisfies original Huber gradient criterion",
            "engine": "whitened-LBFGSB-v04", "initial_shortcut": True}
    initial = (lower.T @ ridge).ravel()
    internal_gtol = min(1e-9, tolerance * n / (4 * np.linalg.norm(lower, ord=np.inf)))
    iterations = evaluations = 0
    attempt_records = []
    for attempt in (1, 2):
        result = minimize(whitened_huber_value_grad, initial.copy(),
            args=(lower, residual, lam, delta), jac=True, method="L-BFGS-B",
            options={"maxiter": 2000 * attempt, "maxfun": 10000 * attempt,
                     "ftol": 0., "gtol": internal_gtol, "maxcor": 20, "maxls": 50})
        z = result.x.reshape(residual.shape)
        g = solve_triangular(lower.T, z, lower=False)
        value, grad = huber_value_grad(g.ravel(), k, residual, lam, delta)
        norm = maxabs(grad)
        accepted = bool(np.isfinite(g).all() and norm <= tolerance)
        iterations += int(result.nit)
        evaluations += int(result.nfev)
        attempt_records.append({"attempt": attempt, "iterations": int(result.nit),
            "message": str(result.message), "grad_inf_original": norm,
            "accepted": accepted})
        if accepted:
            break
    return g, {"accepted": accepted, "seconds": time.perf_counter() - start,
        "iterations": iterations, "evaluations": evaluations, "attempts": attempt,
        "grad_inf": norm, "grad_tolerance": tolerance, "objective": value,
        "message": str(result.message), "engine": "whitened-LBFGSB-v04",
        "initial_shortcut": False, "internal_gtol_unscaled": internal_gtol,
        "attempts_detail": attempt_records}
