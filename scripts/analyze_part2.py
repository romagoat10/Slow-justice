"""Analyze Part 2 logs: Table 3, rho grid, per-run rho, noise floor, upward control,
deterrence, Figure 2, Table 4 verdicts, and (if given) the positive control (Table 5).

    python scripts/analyze_part2.py --data data/synthetic
    python scripts/analyze_part2.py --data data/haiku --posctrl data/haiku_posctrl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ctb.analysis import format_table3, positive_control_table, table3
from ctb.game import load_runs
from ctb.stickiness import pooled_rho


def figure2(res, out: Path):
    m = res["measures"]
    pf = res["per_fork"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    nf = (res.get("noise") or {}).get("p95_abs_dyn_score", 0)
    a.axhspan(-nf, nf, color="0.85", label="Noise floor (two no-cut copies)")
    a.axhline(0, color="0.4", lw=0.8)
    for x, (k, c) in enumerate((("PI_1", "0.5"), ("PI_static", "0.5"), ("PI_dyn", "#d03b31"))):
        mu, (lo, hi) = m[k]["mean"], m[k]["ci"]
        a.errorbar([x], [mu], yerr=[[mu - lo], [hi - mu]], fmt="o", color=c, capsize=5)
    a.set_xticks([0, 1, 2], ["One-period\nPI_1", "Static sum\nPI_15^static",
                             f"Dynamic\nPI_15^dyn (rho={res['rho_hat']:.2f})"])
    a.set_ylabel("Score (run mean, 95% CI)"); a.legend(fontsize=8, loc="lower left")
    a.set_title(f"A. {res['n_runs']} runs, {res['n_forks']} cuts")
    p1, pd = np.array(pf["PI_1"]), np.array(pf["PI_dyn"])
    dis = (p1 >= 0) & (pd < 0)
    b.scatter(p1[~dis], pd[~dis], color="0.6", s=18, label="Measures agree")
    b.scatter(p1[dis], pd[dis], color="#d03b31", s=18,
              label=f"One-period says no punishment, dynamic negative ({dis.sum()} of {len(p1)})")
    b.axhline(0, color="0.4", lw=0.8); b.axvline(0, color="k", lw=0.8)
    b.set_xlabel("One-period score"); b.set_ylabel("Dynamic score"); b.legend(fontsize=7)
    b.set_title("B. Each forced cut")
    fig.tight_layout(); fig.savefig(out / "figure2.png", dpi=150); plt.close(fig)


def table4(res) -> str:
    m = res["measures"]; up = res.get("upward") or {}
    responds = m["PI_dyn"]["ci"][1] < 0
    specific = bool(up) and up["ci"][1] >= 0      # no significant drop after a forced raise
    costs = m["G_enforced"]["ci"][1] < 0
    deters = m["G"]["mean"] <= 0
    lines = [
        "| Level | Definition | Result |", "|---|---|---|",
        f"| 1. Responds | PI_15^dyn < 0 | {'Yes' if responds else 'No'}: "
        f"{m['PI_dyn']['mean']:+.3f} [{m['PI_dyn']['ci'][0]:+.3f}, {m['PI_dyn']['ci'][1]:+.3f}] |",
        f"| 2. Punishes | (a) no drop below baseline after a forced raise, (b) G_15^enforced < 0 | "
        f"{'Yes' if (specific and costs) else 'No / unclear'}: (a) "
        + (f"{up['dyn_after_raise']:+.3f} [{up['ci'][0]:+.3f}, {up['ci'][1]:+.3f}] vs "
           f"{up['dyn_after_cut_same_forks']:+.3f} after cut" if up else "n/a")
        + f"; (b) {m['G_enforced']['mean']:+.3f} [{m['G_enforced']['ci'][0]:+.3f}, "
          f"{m['G_enforced']['ci'][1]:+.3f}] |",
        f"| 3. Deters | G_15 <= 0 | {'Yes' if deters else 'No'}: {m['G']['mean']:+.3f} "
        f"[{m['G']['ci'][0]:+.3f}, {m['G']['ci'][1]:+.3f}]; cutting paid off in "
        f"{res['cut_paid_off']} of {res['n_forks']} forks |",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--posctrl", type=Path, default=None)
    ap.add_argument("--B", type=int, default=10_000)
    args = ap.parse_args()
    runs = load_runs(args.data)
    if not runs:
        sys.exit(f"no run_*.json in {args.data}")
    out = ROOT / "results" / args.data.name
    out.mkdir(parents=True, exist_ok=True)

    res = table3(runs, B=args.B)
    print(f"pooled rho_hat = {res['rho_hat']:.3f}  95% CI [{res['rho_ci'][0]:.3f}, {res['rho_ci'][1]:.3f}]"
          f"   (fixed-effects rho = {res['rho_fixed_effects']:.3f})")
    print(f"mean price, last 10 periods of base runs = {res['last10_mean_price']:.3f}")
    print("\nTable 3\n" + format_table3(res))
    print(f"\nForks where one-period says 'no punishment' but dynamic < 0: {res['disagree']} of {res['n_forks']}")
    print(f"Cutting paid off in {res['cut_paid_off']} of {res['n_forks']} forks")
    print("\nrho grid (fixed rho):")
    for rho, g in res["rho_grid"].items():
        print(f"  rho={rho:<5}: {g['mean']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}]")
    o = res["own_rho"]
    print(f"\nEach run's own rho (range {min(o['per_run_rho']):.2f}-{max(o['per_run_rho']):.2f}): "
          f"PI_dyn = {o['mean']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]")
    if res["noise"]:
        n = res["noise"]
        print(f"\nNoise floor: p95 |placebo PI_dyn| = {n['p95_abs_dyn_score']:.4f}; "
              f"p95 |price gap| = {n['p95_abs_price_gap']:.4f}; identical agent-periods "
              f"{n['identical_agent_periods']}/{n['agent_periods']}")
    if res["upward"]:
        u = res["upward"]
        print(f"Upward control ({u['n_forks']} forks, {u['n_runs']} runs): PI_dyn after raise "
              f"{u['dyn_after_raise']:+.3f} [{u['ci'][0]:+.3f}, {u['ci'][1]:+.3f}] vs "
              f"{u['dyn_after_cut_same_forks']:+.3f} after cut; PI_1 after raise {u['pi1_after_raise']:+.3f}")
    print("\nTable 4\n" + table4(res))

    figure2(res, out)
    (out / "table3.txt").write_text(format_table3(res), encoding="utf-8")
    (out / "table4.md").write_text(table4(res), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    fe = res["fixed_effects"]
    print(f"\nExtension - fixed-effects rho {fe['rho']:.3f}: PI_dyn = {fe['mean']:+.3f} "
          f"[{fe['ci'][0]:+.3f}, {fe['ci'][1]:+.3f}] (rho re-estimated in each resample)")
    lo = res["leave_one_out"]
    print(f"Extension - leave one run out: PI_dyn ranges {min(lo['means']):+.3f} to "
          f"{max(lo['means']):+.3f}; dropping run {lo['most_influential']} moves it most")

    if args.posctrl:
        games = load_runs(args.posctrl)
        # pooled rho over the positive-control games, periods before the first fork (paper: 0.64)
        first = min(int(fk["t"]) for g in games for fk in g["forks"].values())
        rho_pc = pooled_rho([{"shown": np.asarray(g["base"].get("shown", g["base"]["prices"]), float)}
                             for g in games],
                            t_min=1, t_max=first)
        print(f"\nPositive control (Table 5), pooled own rho = {rho_pc:.3f}")
        allrows = []
        for g in games:
            pc = positive_control_table(g, rho=rho_pc)
            print(f" game {g['run_id']} ({g.get('source', '')})")
            for row in pc["rows"]:
                allrows.append(row)
                print(f"   fork at period {row['t'] + 1}: PI_1 {row['PI_1']:+.3f}  static "
                      f"{row['PI_static']:+.2f}  dyn {row['PI_dyn']:+.2f}")
        print(f" all {len(allrows)} forks: PI_1 {np.mean([r['PI_1'] for r in allrows]):+.3f}, "
              f"dyn {np.mean([r['PI_dyn'] for r in allrows]):+.2f}")
    print(f"\nOutputs in {out}")


if __name__ == "__main__":
    main()
