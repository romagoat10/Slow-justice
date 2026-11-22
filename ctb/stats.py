"""Statistics used in Section 3.5: run-level bootstrap and a two-sided sign test."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def sign_test_p(values: Sequence[float]) -> tuple[int, int, float]:
    """Two-sided exact binomial sign test of 'median = 0'. Zeros are dropped.

    Returns (n_below_zero, n_nonzero, p). Paper checks: 0/12 -> 0.0005, 9/12 -> 0.146,
    6/12 -> 1.00, 1/12 -> 0.006.
    """
    v = [x for x in values if x != 0]
    n, k = len(v), sum(1 for x in v if x < 0)
    if n == 0:
        return 0, 0, 1.0
    tail = min(k, n - k)
    p = 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n
    return k, n, min(1.0, p)


def bootstrap_runs(n_runs: int, stat: Callable[[np.ndarray], float], B: int = 10_000,
                   seed: int = 0, alpha: float = 0.05) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap over runs.

    `stat(idx)` receives an array of resampled run indices (with replacement) and returns
    the statistic. Doing it this way lets the caller re-estimate rho inside each resample,
    as the paper does.
    """
    rng = np.random.default_rng(seed)
    draws = np.empty(B)
    for b in range(B):
        draws[b] = stat(rng.integers(0, n_runs, n_runs))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), draws
