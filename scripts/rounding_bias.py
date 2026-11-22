"""Extension experiment (not in the paper): does two-decimal price rounding bias rho_hat and
PI_dyn for a sticky agent that NEVER punishes?

LLM agents output prices like 2.07, so a move smaller than half a cent is invisible.
With rho = 0.92 the agent moves 8% of its gap per period, so it stops moving once the gap
is under ~0.0625 (the "dead band"). That violates the linear partial-adjustment model the
dynamic baseline assumes. Small sampling noise "dithers" the rounding and partly restores
linearity. This script measures the bias as a function of noise.

    python scripts/rounding_bias.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ctb.agents import SyntheticAgent
from ctb.analysis import _cache
from ctb.game import FORK_PERIODS, H, base_run, fork_slice, run_fork
from ctb.stickiness import fixed_effects_rho, pooled_rho

TRUE_RHO = 0.92


def simulate(noise, delta=0.0, runs=12, seed0=0):
    out = []
    for r in range(runs):
        ag = tuple(SyntheticAgent(rho=TRUE_RHO, delta=delta, noise_sd=noise,
                                  seed=seed0 + 10 * r + i) for i in (1, 2))
        base = base_run(ag, 150)
        forks = {}
        for k, t in enumerate(FORK_PERIODS):
            fk = {"t": t, "cutter": k % 2}
            for kind in ("cut", "nocut", "nocut2"):
                fk[kind] = fork_slice(run_fork(base, ag, t, k % 2, kind, H), t, H)
            forks[str(t)] = fk
        out.append({"base": base.to_json(), "forks": forks})
    return out


def summarize(runs):
    shown = [{"shown": np.asarray(r["base"]["prices"], float)} for r in runs]
    rho_hat = pooled_rho(shown)
    fcs = _cache(runs)
    same = np.mean([np.isclose(np.asarray(r["forks"][k]["nocut"]),
                               np.asarray(r["forks"][k]["nocut2"])).mean()
                    for r in runs for k in r["forks"]])
    return dict(rho_hat=rho_hat, rho_fe=fixed_effects_rho(shown),
                dyn_at_hat=np.mean([fc.dyn(rho_hat) for fc in fcs]),
                dyn_at_true=np.mean([fc.dyn(TRUE_RHO) for fc in fcs]),
                identical=same)


if __name__ == "__main__":
    print(f"Sticky NON-punisher, true rho = {TRUE_RHO}. A correct measure should give PI_dyn ~ 0.\n")
    print(f"{'noise sd':>9} {'copies identical':>17} {'rho_hat':>8} {'rho_FE':>7} "
          f"{'PI_dyn @rho_hat':>16} {'PI_dyn @true rho':>17}")
    for noise in (0.0, 0.0005, 0.0012, 0.002, 0.004, 0.008):
        s = summarize(simulate(noise))
        print(f"{noise:9.4f} {s['identical']:17.1%} {s['rho_hat']:8.3f} {s['rho_fe']:7.3f} "
              f"{s['dyn_at_hat']:+16.3f} {s['dyn_at_true']:+17.3f}")
    print("\nSame grid with a true punisher (delta = 0.10, K = 10; unrounded theory: -0.571):")
    for noise in (0.0, 0.0012, 0.004):
        s = summarize(simulate(noise, delta=0.10))
        print(f"  noise {noise:.4f}: rho_hat {s['rho_hat']:.3f}, PI_dyn @rho_hat {s['dyn_at_hat']:+.3f}, "
              f"@true {s['dyn_at_true']:+.3f}")
