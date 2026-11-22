"""Estimating an agent's stickiness rho from its ordinary price history (Section 3.4).

Paper regression, pooled over both agents and all runs, periods 1-75, using only the
two-decimal prices the agents were shown:

    p_B(t) - BR(q_shown(t-1)) = a + rho * [ p_B(t-1) - BR(q_shown(t-1)) ] + eps

Interpretation: y is "how far my price is from my static best response to what I saw";
x is the same gap one period earlier. rho near 1 = the agent barely closes the gap.

Also provided (not in the paper, see README "Critical notes"):
    * per-run estimates
    * a run-fixed-effects ("within") estimator, which removes the between-run level
      differences that can inflate a pooled-intercept rho.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .market import best_response


def regression_rows(own: Sequence[float], rival_shown: Sequence[float],
                    t_min: int = 1, t_max: int = 75) -> tuple[np.ndarray, np.ndarray]:
    """Build (x, y) for one agent in one run. own/rival_shown are indexed by period 0..T."""
    xs, ys = [], []
    for t in range(max(t_min, 1), min(t_max, len(own) - 1) + 1):
        br = best_response(float(rival_shown[t - 1]))
        ys.append(own[t] - br)
        xs.append(own[t - 1] - br)
    return np.asarray(xs), np.asarray(ys)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (intercept a, slope rho)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0]), float(beta[1])


def run_rows(run: dict, t_min=1, t_max=75) -> list[tuple[np.ndarray, np.ndarray]]:
    """Both agents of one base run. `run` must contain 'shown' as a (T+1, 2) array."""
    shown = np.asarray(run["shown"], float)
    return [regression_rows(shown[:, i], shown[:, 1 - i], t_min, t_max) for i in (0, 1)]


def pooled_rho(runs: Iterable[dict], t_min=1, t_max=75) -> float:
    xs, ys = [], []
    for run in runs:
        for x, y in run_rows(run, t_min, t_max):
            xs.append(x); ys.append(y)
    return ols(np.concatenate(xs), np.concatenate(ys))[1]


def per_run_rho(run: dict, t_min=1, t_max=75) -> float:
    rows = run_rows(run, t_min, t_max)
    return ols(np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]))[1]


def fixed_effects_rho(runs: Iterable[dict], t_min=1, t_max=75) -> float:
    """Within estimator: demean x and y inside each (run, agent) series before pooling."""
    xs, ys = [], []
    for run in runs:
        for x, y in run_rows(run, t_min, t_max):
            if len(x) > 1:
                xs.append(x - x.mean()); ys.append(y - y.mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    denom = float(x @ x)
    return float(x @ y) / denom if denom > 0 else float("nan")
