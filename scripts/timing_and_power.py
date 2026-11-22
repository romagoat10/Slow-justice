"""(1) WHEN does Haiku respond? Average gap to the dynamic baseline, d(t+h) - d_hat(t+h), for each
h = 1..15 over the 48 forks, with a run-level bootstrap band. Writes results/haiku/figure5_timing.png.

(2) HOW MANY RUNS would a follow-up need? Resamples the 12 observed run means to N runs and
reports how often the 95% interval on PI_dyn is below zero and how often the run-level sign
test is significant (p < 0.05). This treats the 12 runs as the population, so it is a rough guide.

    python scripts/timing_and_power.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ctb.analysis import _cache, _shown
from ctb.game import load_runs
from ctb.measures import dynamic_baseline
from ctb.stats import sign_test_p
from ctb.stickiness import pooled_rho

H = 15


def timing(runs, rho, B=5000, seed=0):
    forks = _cache(runs)
    gap = np.array([fc.d[1:H + 1] - dynamic_baseline(fc.dBR, rho, H)[1:H + 1] for fc in forks])
    instant = np.array([fc.d[1:H + 1] - fc.dBR[0:H] for fc in forks])
    rid = np.array([fc.run for fc in forks]); n = len(runs)
    run_gap = np.array([gap[rid == r].mean(0) for r in range(n)])
    rng = np.random.default_rng(seed)
    boot = np.array([run_gap[rng.integers(0, n, n)].mean(0) for _ in range(B)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    mean = run_gap.mean(0)
    h = np.arange(1, H + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.axhline(0, color="0.4", lw=0.8)
    a.fill_between(h, lo, hi, color="#d03b31", alpha=0.18, label="95% band (runs resampled)")
    a.plot(h, mean, "o-", color="#d03b31", ms=4, label="Haiku: d − d̂ (vs sticky non-punisher)")
    a.plot(h, np.array([instant[rid == r].mean(0) for r in range(n)]).mean(0), "s--", color="0.45", ms=3,
           label="d − ΔBR (vs instant best response)")
    a.set_xlabel("Periods after the forced cut"); a.set_ylabel("Rival price gap")
    a.set_title("A. Per-period response", fontsize=10); a.legend(fontsize=7)
    cum, clo, chi = np.cumsum(mean), np.quantile(np.cumsum(boot, 1), 0.025, 0), np.quantile(np.cumsum(boot, 1), 0.975, 0)
    b.axhline(0, color="0.4", lw=0.8)
    b.fill_between(h, clo, chi, color="#d03b31", alpha=0.18)
    b.plot(h, cum, "o-", color="#d03b31", ms=4)
    b.set_xlabel("Horizon H"); b.set_ylabel("Cumulative PI_H^dyn")
    b.set_title("B. Cumulative dynamic score", fontsize=10)
    fig.tight_layout()
    out = ROOT / "results" / "haiku" / "figure5_timing.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    first_sig = next((int(x) for x, u in zip(h, chi) if u < 0), None)
    print("1. Timing: mean d - d_hat by period after the cut (run-level 95% band)")
    for k in range(H):
        print(f"   h={k + 1:2d}: {mean[k]:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]   cumulative {cum[k]:+.3f}")
    print(f"   h=1 gap: {mean[0]:+.3f}; largest single-period gap at h={int(np.argmin(mean)) + 1} ({mean.min():+.3f})")
    print(f"   cumulative score first has its whole 95% band below zero at H = {first_sig}")
    print(f"   figure: {out}")
    return dict(mean=mean.tolist(), lo=lo.tolist(), hi=hi.tolist(), cum=cum.tolist(), first_sig=first_sig)


def power(run_means, Ns=(12, 24, 36, 48), sims=5000, seed=1):
    rng = np.random.default_rng(seed)
    print("\n2. Power: resampling the 12 observed run means to N runs")
    out = {}
    for N in Ns:
        ci_ok = sign_ok = 0
        for _ in range(sims):
            x = run_means[rng.integers(0, len(run_means), N)]
            se = x.std(ddof=1) / np.sqrt(N)
            ci_ok += (x.mean() + 1.96 * se) < 0
            k, nz, p = sign_test_p(x)
            sign_ok += (p < 0.05) and k > nz / 2
        out[N] = (ci_ok / sims, sign_ok / sims)
        print(f"   N = {N:2d} runs: interval below zero {ci_ok / sims:5.0%}, sign test significant {sign_ok / sims:5.0%}")
    return out


if __name__ == "__main__":
    runs = load_runs(ROOT / "data" / "haiku")
    rho = pooled_rho([_shown(r) for r in runs])
    t = timing(runs, rho)
    res = json.loads((ROOT / "results" / "haiku" / "results.json").read_text(encoding="utf-8"))
    p_pooled = power(np.array(res["measures"]["PI_dyn"]["per_run"]))
    print("   (same, at the fixed-effects rho)")
    fc = _cache(runs); rid = np.array([f.run for f in fc])
    fe = np.array([np.mean([f.dyn(res["fixed_effects"]["rho"]) for f in fc if f.run == r]) for r in range(len(runs))])
    p_fe = power(fe)
    (ROOT / "results" / "haiku" / "timing_power.json").write_text(json.dumps(
        {"timing": t, "power_pooled": p_pooled, "power_fe": p_fe}, indent=2), encoding="utf-8")
