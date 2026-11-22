"""Logit Bertrand duopoly used in the paper (Calvano et al. 2020; Fish et al. 2024).

Paper, Section 3.1:
    a = 2, a_0 = 0, mu = 0.25, marginal cost c = 1.
    One-shot Nash price            ~ 1.473
    Joint monopoly price           ~ 1.925
    Single-firm, single-product monopoly price ~ 1.802
    BR(q) = price that maximizes a firm's profit when its rival charges q.

Demand for firm i (with an outside good of quality a_0):

    s_i = exp((a - p_i)/mu) / ( exp((a - p_i)/mu) + exp((a - p_j)/mu) + exp(a_0/mu) )

Everything here is pure numpy / stdlib so it runs without scipy.
"""
from __future__ import annotations

import math
from functools import lru_cache

A = 2.0      # product quality (both firms)
A0 = 0.0     # outside-good quality
MU = 0.25    # horizontal differentiation
C = 1.0      # marginal cost

# Fish et al. scale quantities so that agents see "units sold". It does not change any
# price-based measure; it only scales profits (and therefore G_H) linearly.
QTY_SCALE = 1.0


def demand(p_i: float, p_j: float) -> float:
    """Market share of firm i when it charges p_i and the rival charges p_j."""
    e_i = math.exp((A - p_i) / MU)
    e_j = math.exp((A - p_j) / MU)
    e_0 = math.exp(A0 / MU)
    return QTY_SCALE * e_i / (e_i + e_j + e_0)


def profit(p_i: float, p_j: float) -> float:
    return (p_i - C) * demand(p_i, p_j)


def monopoly_demand(p: float) -> float:
    """One firm, one product, only the outside good competes (Fish et al. monopoly stage)."""
    e = math.exp((A - p) / MU)
    return QTY_SCALE * e / (e + math.exp(A0 / MU))


def monopoly_profit(p: float) -> float:
    return (p - C) * monopoly_demand(p)


def _bisect(f, lo: float, hi: float, tol: float = 1e-13, it: int = 300) -> float:
    flo = f(lo)
    if flo * f(hi) > 0:
        raise ValueError("root not bracketed")
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(hi - lo) < tol:
            break
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


@lru_cache(maxsize=200_000)
def best_response(q: float) -> float:
    """BR(q): argmax_p (p - c) * s(p, q).

    Logit FOC:  p - c = mu / (1 - s(p, q)).   The LHS-RHS is strictly increasing in p,
    so a bisection on [c, c + 10] is safe.
    """
    def foc(p):
        s = demand(p, q) / QTY_SCALE
        return (p - C) - MU / (1.0 - s)
    return _bisect(foc, C + 1e-9, C + 10.0)


def nash_price() -> float:
    """Symmetric one-shot Nash: p = BR(p)."""
    return _bisect(lambda p: best_response(p) - p, C + 1e-6, C + 5.0)


def joint_monopoly_price() -> float:
    """Symmetric price maximizing joint profit. FOC: p - c = mu / (1 - 2 s(p, p))."""
    def foc(p):
        s = demand(p, p) / QTY_SCALE
        return (p - C) - MU / (1.0 - 2.0 * s)
    return _bisect(foc, C + 1e-6, C + 5.0)


def single_product_monopoly_price() -> float:
    """One product vs the outside good. FOC: p - c = mu / (1 - s)."""
    def foc(p):
        s = monopoly_demand(p) / QTY_SCALE
        return (p - C) - MU / (1.0 - s)
    return _bisect(foc, C + 1e-6, C + 5.0)


if __name__ == "__main__":
    print(f"Nash price               {nash_price():.4f}   (paper: 1.473)")
    print(f"Joint monopoly price     {joint_monopoly_price():.4f}   (paper: 1.925)")
    print(f"Single-product monopoly  {single_product_monopoly_price():.4f}   (paper: 1.802)")
    print(f"BR(1.90)                 {best_response(1.90):.4f}")
    print(f"BR(BR(1.90)) - BR(1.90)  {best_response(best_response(1.90)) - best_response(1.90):+.4f}")
