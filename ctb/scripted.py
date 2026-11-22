"""Part 1 of the paper: scripted rivals where the right answer is known (Section 3.3).

Rival rule:
    p_B(t) = rho * p_B(t-1) + (1 - rho) * target(t)
    target(t) = m + BR(q_shown(t-1))        minus delta for K = 10 periods after it sees a cut
    m = 1.90 - BR(1.90)                      (holds B at 1.90 when nothing happens)

Cutter: cuts once, at the fork period, to BR(1.90) (~1.6526).
    "returns"   : back to 1.90 the next period (Tran & Yu design)
    "stays low" : keeps charging BR(1.90) for the whole horizon

"Sees a cut" = the shown rival price fell relative to the previous shown price. When the
cutter stays low there is only one such drop, so the punishment fires once. This matches
every number in Table 2 (the punishment contribution is identical in both cutter versions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import best_response
from .measures import Fork, score_all

P0 = 1.90
K_PUNISH = 10
H_DEFAULT = 15


@dataclass
class ScriptedRival:
    rho: float
    delta: float
    m: float
    K: int = K_PUNISH

    def run(self, q_path: np.ndarray, pB0: float) -> np.ndarray:
        """Return B's prices for h = 0..H given the cutter's shown prices q_path[0..H]."""
        H = len(q_path) - 1
        pB = np.empty(H + 1)
        pB[0] = pB0
        punish_left = 0
        prev_q = P0
        for h in range(1, H + 1):
            q = q_path[h - 1]            # what B saw last period
            if q < prev_q - 1e-12:       # a cut became visible
                punish_left = self.K
            prev_q = q
            target = self.m + best_response(q)
            if punish_left > 0:
                target -= self.delta
                punish_left -= 1
            pB[h] = self.rho * pB[h - 1] + (1 - self.rho) * target
        return pB


def make_fork(rho: float, delta: float, cutter: str = "returns", H: int = H_DEFAULT,
              p0: float = P0) -> Fork:
    m = p0 - best_response(p0)
    cut = best_response(p0)
    pA_nocut = np.full(H + 1, p0)
    if cutter == "returns":
        pA_cut = np.r_[cut, np.full(H, p0)]
    elif cutter == "stays low":
        pA_cut = np.full(H + 1, cut)
    else:
        raise ValueError(cutter)
    rival = ScriptedRival(rho, delta, m)
    pB_cut = rival.run(pA_cut, p0)
    pB_nocut = rival.run(pA_nocut, p0)
    return Fork(pA_cut, pA_nocut, pB_cut, pB_nocut,
                meta=dict(rho=rho, delta=delta, cutter=cutter))


def table2(H: int = H_DEFAULT) -> list[dict]:
    rows = []
    for cutter, rhos in (("returns", (0, 0.5, 0.8, 0.92)), ("stays low", (0.5, 0.8, 0.92))):
        for rho in rhos:
            for delta in (0.0, 0.10):
                f = make_fork(rho, delta, cutter, H)
                s = score_all(f, rho=rho, H=H)   # dynamic baseline at the TRUE rho
                rows.append(dict(rho=rho, delta=delta, cutter=cutter,
                                 punishes=delta > 0, **s))
    return rows


# Table 2 exactly as printed in the paper, for automated comparison.
PAPER_TABLE2 = {
    # (rho, delta, cutter): (PI_1, static, dynamic)
    (0, 0.0, "returns"): (0.000, 0.000, 0.000),
    (0, 0.1, "returns"): (-0.100, -1.000, -1.000),
    (0.5, 0.0, "returns"): (0.051, 0.000, 0.000),
    (0.5, 0.1, "returns"): (0.001, -0.997, -0.997),
    (0.8, 0.0, "returns"): (0.081, 0.004, 0.000),
    (0.8, 0.1, "returns"): (0.061, -0.879, -0.883),
    (0.92, 0.0, "returns"): (0.094, 0.029, 0.000),
    (0.92, 0.1, "returns"): (0.086, -0.542, -0.571),
    (0.5, 0.0, "stays low"): (0.051, 0.102, 0.000),
    (0.5, 0.1, "stays low"): (0.001, -0.895, -0.997),
    (0.8, 0.0, "stays low"): (0.081, 0.393, 0.000),
    (0.8, 0.1, "stays low"): (0.061, -0.490, -0.883),
    (0.92, 0.0, "stays low"): (0.094, 0.836, 0.000),
    (0.92, 0.1, "stays low"): (0.086, 0.265, -0.571),
}


def wrong_rho_sensitivity(true_rho: float = 0.92, cutter: str = "stays low",
                          rhos=None, H: int = H_DEFAULT) -> list[tuple[float, float]]:
    """Section 4.1 / Figure 4: dynamic score of a NON-punisher when rho is misspecified."""
    if rhos is None:
        rhos = np.round(np.arange(0.0, 0.991, 0.01), 2)
    f = make_fork(true_rho, 0.0, cutter, H)
    from .measures import pi_dynamic
    return [(float(r), pi_dynamic(f, float(r), H)) for r in rhos]


def detection_sweep(delta: float = 0.10, cutter: str = "returns", H: int = H_DEFAULT,
                    rhos=None) -> list[dict]:
    """Extension: where along rho does each measure stop catching a punisher?"""
    if rhos is None:
        rhos = np.round(np.arange(0.0, 0.981, 0.02), 2)
    out = []
    for r in rhos:
        f = make_fork(float(r), delta, cutter, H)
        out.append(dict(rho=float(r), **score_all(f, float(r), H)))
    return out
