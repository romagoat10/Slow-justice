"""Punishment / deterrence measures from Section 3.2 of the paper.

A *fork* at period t has two copies of the market:
    cut copy    : cutter A's period-t price is replaced (e.g. by 0.9 * p_A(t-1))
    no-cut copy : nothing changes
Rival B's period-t price is the same in both copies (it was set before the cut was seen).

Index convention used everywhere in this module (h = periods after the fork):

    pB_cut[h], pB_nocut[h]   h = 0..H     rival price in each copy (pB_*[0] equal)
    pA_cut[h], pA_nocut[h]   h = 0..H     cutter price in each copy
    qshown_cut[h]            h = 0..H     the cutter price B was *shown* for period t+h
                                          (two-decimal rounding for LLM runs)

Definitions (paper notation -> code):

    d(t+h)        = pB_cut[h] - pB_nocut[h]
    dBR(t+h-1)    = BR(qshown_cut[h-1]) - BR(qshown_nocut[h-1])

    PI_1          = d(t+1) - dBR(t)
    PI_H^static   = sum_{h=1..H} [ d(t+h) - dBR(t+h-1) ]
    d_hat(t)      = 0
    d_hat(t+h)    = rho * d_hat(t+h-1) + (1 - rho) * dBR(t+h-1)
    PI_H^dyn      = sum_{h=1..H} [ d(t+h) - d_hat(t+h) ]

    G_H           = sum_{k=0..H} [ pi_A^cut(t+k) - pi_A^nocut(t+k) ]
    G_H^enforced  = sum_{k=0..H} [ pi_A(pA_cut, pB_cut) - pi_A(pA_cut, pB_nocut + d_hat) ]

Negative PI = B priced lower than a rival that does not punish would have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .market import best_response, profit


@dataclass
class Fork:
    """Everything needed to score one forced deviation."""
    pA_cut: Sequence[float]
    pA_nocut: Sequence[float]
    pB_cut: Sequence[float]
    pB_nocut: Sequence[float]
    qshown_cut: Sequence[float] | None = None    # defaults to round(pA_cut, 2)? no: to pA_cut
    qshown_nocut: Sequence[float] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("pA_cut", "pA_nocut", "pB_cut", "pB_nocut"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if self.qshown_cut is None:
            self.qshown_cut = self.pA_cut.copy()
        if self.qshown_nocut is None:
            self.qshown_nocut = self.pA_nocut.copy()
        self.qshown_cut = np.asarray(self.qshown_cut, dtype=float)
        self.qshown_nocut = np.asarray(self.qshown_nocut, dtype=float)

    @property
    def H(self) -> int:
        return len(self.pB_cut) - 1


def d_series(f: Fork) -> np.ndarray:
    """d[h] for h = 0..H (d[0] should be 0)."""
    return f.pB_cut - f.pB_nocut


def dBR_series(f: Fork) -> np.ndarray:
    """dBR[h] = BR(q_cut[h]) - BR(q_nocut[h]) for h = 0..H."""
    return np.array([best_response(float(a)) - best_response(float(b))
                     for a, b in zip(f.qshown_cut, f.qshown_nocut)])


def dynamic_baseline(dBR: np.ndarray, rho: float, H: int) -> np.ndarray:
    """d_hat[h], h = 0..H, for a sticky non-punisher with stickiness rho."""
    dh = np.zeros(H + 1)
    for h in range(1, H + 1):
        dh[h] = rho * dh[h - 1] + (1.0 - rho) * dBR[h - 1]
    return dh


def pi_one_period(f: Fork) -> float:
    d, dBR = d_series(f), dBR_series(f)
    return float(d[1] - dBR[0])


def pi_static(f: Fork, H: int | None = None) -> float:
    H = f.H if H is None else H
    d, dBR = d_series(f), dBR_series(f)
    return float(sum(d[h] - dBR[h - 1] for h in range(1, H + 1)))


def pi_dynamic(f: Fork, rho: float, H: int | None = None) -> float:
    H = f.H if H is None else H
    d, dBR = d_series(f), dBR_series(f)
    dh = dynamic_baseline(dBR, rho, H)
    return float(sum(d[h] - dh[h] for h in range(1, H + 1)))


def deterrence(f: Fork, H: int | None = None) -> float:
    """G_H: the cutter's cumulative profit gain from cutting (k = 0..H, i.e. H+1 periods)."""
    H = f.H if H is None else H
    return float(sum(profit(f.pA_cut[k], f.pB_cut[k]) - profit(f.pA_nocut[k], f.pB_nocut[k])
                     for k in range(0, H + 1)))


def deterrence_enforced(f: Fork, rho: float, H: int | None = None) -> float:
    """G_H^enforced: part of G_H caused by B pricing below the non-punishing baseline.

    Holds A's cut-copy prices fixed and swaps B's cut-copy price for p_B^nocut + d_hat.
    """
    H = f.H if H is None else H
    dh = dynamic_baseline(dBR_series(f), rho, H)
    base_B = f.pB_nocut + dh
    return float(sum(profit(f.pA_cut[k], f.pB_cut[k]) - profit(f.pA_cut[k], base_B[k])
                     for k in range(0, H + 1)))


def score_all(f: Fork, rho: float, H: int = 15) -> dict:
    return {
        "PI_1": pi_one_period(f),
        "PI_static": pi_static(f, H),
        "PI_dyn": pi_dynamic(f, rho, H),
        "G": deterrence(f, H),
        "G_enforced": deterrence_enforced(f, rho, H),
    }
