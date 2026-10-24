"""Label-independent, jointly bounded per-support kernel coefficients."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .solvers.core import regularized_kernel, kernel, projected_gradient, objective
from .solvers.core_v04 import BoxWorkspace
from .solvers.core import solve_box as early_solve_box


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if x.ndim != 2 or not np.isfinite(x).all() or np.any(norms == 0):
        raise ValueError('finite, nonzero feature rows are required')
    return x / norms


def coefficients(support, query, config, *, stage, support_targets=False):
    s, q = unit_rows(support), unit_rows(query)
    gram, jitter = regularized_kernel(s, config['beta'])
    a = gram + float(config['lambda']) * np.eye(len(s))
    cross = gram if support_targets else kernel(s, q, config['beta'])
    linear = cho_solve(cho_factor(a, lower=True), cross)
    tau = float(config['tau'])
    if stage not in ('text', 'final'):
        raise ValueError('geometry stage must be text or final')
    workspace = BoxWorkspace(a, tau) if stage == 'final' else None
    columns, records = [], []
    for j in range(len(q)):
        k = cross[:, j]
        result = workspace.solve(k, linear[:, j]) if workspace else early_solve_box(a, k, tau, linear[:, j])
        w = result.w.copy()
        if not result.accepted and stage == 'text':
            free = abs(w) < tau - 1e-10
            fixed = ~free
            candidate = w.copy()
            candidate[free] = np.linalg.solve(a[np.ix_(free, free)], k[free] - a[np.ix_(free, fixed)] @ w[fixed])
            if np.max(abs(candidate), initial=0.) <= tau and projected_gradient(candidate, a, k, tau) <= result.pg_tolerance:
                w = candidate
            else:
                for _ in range(10000):
                    for i in range(len(s)):
                        w[i] = np.clip(w[i] + (k[i] - a[i] @ w) / a[i, i], -tau, tau)
                    if projected_gradient(w, a, k, tau) <= result.pg_tolerance:
                        break
        pg = projected_gradient(w, a, k, tau)
        accepted = (np.isfinite(w).all() and pg <= result.pg_tolerance
                    and np.max(abs(w), initial=0.) <= tau + 1e-10
                    and objective(w, a, k) <= result.clip_objective + 1e-8 * (1 + abs(result.clip_objective)))
        if not accepted:
            raise RuntimeError(f'Kernel solve failed the recorded numerical checks at query {j}')
        columns.append(w)
        records.append(float(pg / result.pg_tolerance))
    return np.stack(columns, axis=1), {'jitter':jitter, 'max_pg_ratio':max(records, default=0.), 'stage':stage}
