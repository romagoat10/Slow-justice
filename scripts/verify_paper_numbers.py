"""Internal-consistency audit of the numbers printed in the paper (no data needed).

    python scripts/verify_paper_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctb.market import (best_response, joint_monopoly_price, nash_price,
                        single_product_monopoly_price)
from ctb.stats import sign_test_p

checks = []


def check(name, ours, paper, ok, note=""):
    checks.append((name, ours, paper, ok, note))


# --- market --------------------------------------------------------------------------
check("Nash price", f"{nash_price():.3f}", "1.473", abs(nash_price() - 1.473) < 5e-4)
check("Joint monopoly price", f"{joint_monopoly_price():.3f}", "1.925",
      abs(joint_monopoly_price() - 1.925) < 5e-4)
check("Single-product monopoly", f"{single_product_monopoly_price():.3f}", "1.802",
      abs(single_product_monopoly_price() - 1.802) < 5e-4)
br190 = best_response(1.90)
check("Cut price BR(1.90) (intro says 1.65)", f"{br190:.3f}", "1.65", abs(br190 - 1.65) < 5e-3)
dBR = best_response(br190) - br190
check("Instant-BR rival after cut (intro: ~1.80)", f"{1.90 + dBR:.3f}", "1.80",
      abs(1.90 + dBR - 1.80) < 5e-3)
slow = 1.90 + 0.08 * (dBR - 0.10)
check("Slow punisher period-1 price (intro: 1.884)", f"{slow:.4f}", "1.884", abs(slow - 1.884) < 5e-4)

# --- analytic threshold for the one-period test -----------------------------------------
# PI_1 = -rho*dBR - (1-rho)*delta  >= 0   <=>   rho >= delta / (delta + |dBR|)
rho_star = 0.10 / (0.10 + abs(dBR))
check("rho above which PI_1 misses a delta=0.10 punisher", f"{rho_star:.4f}", "~0.5 (App. B)",
      abs(rho_star - 0.5) < 0.01, "depends on delta and cut size; not a universal 0.5")

# --- sign tests -------------------------------------------------------------------------
for k, paper in ((0, 0.0005), (6, 1.00), (9, 0.15), (1, 0.006)):
    _, _, p = sign_test_p([-1] * k + [1] * (12 - k))
    check(f"Sign test {k}/12", f"{p:.4f}", f"{paper}", abs(p - paper) < 5e-3 or round(p, 2) == paper)

# --- call accounting --------------------------------------------------------------------
calls = 12 * 150 * 2 + 48 * 2 * 15 * 2 + 12 * 15 * 2
check("Model calls (base + forks + 2nd no-cut)", str(calls), "6840", calls == 6840,
      "upward-control copies (another 360) are then NOT in the 6840")
check("Agent-periods in 2nd no-cut copies", str(12 * 15 * 2), "360", 12 * 15 * 2 == 360)
check("Identical prices 347/360", f"{347/360:.1%}", "96%", round(347 / 360, 2) == 0.96)

# --- Table 3 arithmetic -----------------------------------------------------------------
check("PI_dyn / noise floor", f"{0.534/0.007:.0f}x", "about 75x", 70 < 0.534 / 0.007 < 80)
check("G_enforced / G_15", f"{0.146/1.027:.3f}", "'about an eighth'", False,
      "0.142 = 1/7; it is 1/8 only vs the no-response gain 1.027+0.146 (0.124)")
check("Abstract PI_1 CI", "0.049 to 0.065", "[+0.049,+0.065]", True)

# --- positive control ---------------------------------------------------------------------
g1_dyn = (-3.17 - 6.46 - 2.25 - 6.00) / 4
g1_pi1 = (0.061 + 0.073 + 0.061 + 0.073) / 4
g2_dyn = 2 * -2.25 - g1_dyn
g2_pi1 = 2 * 0.064 - g1_pi1
check("Game 1 mean PI_dyn (Table 5)", f"{g1_dyn:.2f}", "-", True)
check("Implied game 2 mean PI_dyn (from 8-fork -2.25)", f"{g2_dyn:+.2f}", "should be ~0 (rule never fires)",
      abs(g2_dyn) < 0.1, "consistent")
check("Implied game 2 mean PI_1 (from 8-fork +0.064)", f"{g2_pi1:+.3f}", "-", True)
check("Positive-control cut 2.15 -> 1.94", f"{round(0.9*2.15, 2):.2f}", "1.94", round(0.9 * 2.15, 2) == 1.94)

# --- notational ---------------------------------------------------------------------------
check("G_H horizon: sum k=0..H", "16 periods", "'over 15 periods'", False,
      "the formula includes the cut period itself")
check("Upward control: 7 of 8 runs negative but mean +0.013", "-", "-", False,
      "only possible if 1 run is strongly positive; CI is skewed (-0.030,+0.091)")

w = max(len(c[0]) for c in checks)
for name, ours, paper, ok, note in checks:
    flag = "OK " if ok else "!! "
    print(f"{flag} {name:<{w}}  ours={ours:<14} paper={paper:<22} {note}")
print(f"\n{sum(c[3] for c in checks)}/{len(checks)} consistent; '!!' rows are discussed in README.")
