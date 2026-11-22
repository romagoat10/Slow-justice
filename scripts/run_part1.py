"""Part 1: scripted rivals. Reproduces Table 2, Figure 1, Figure 4 (wrong rho) and adds a
detection sweep over rho. No API calls; runs in a second.

    python scripts/run_part1.py
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

from ctb.measures import dBR_series, dynamic_baseline
from ctb.scripted import (PAPER_TABLE2, detection_sweep, make_fork, table2,
                          wrong_rho_sensitivity)

OUT = ROOT / "results" / "part1"
OUT.mkdir(parents=True, exist_ok=True)


def print_table2():
    rows = table2()
    lines = ["| rho | delta | cutter | punishes | one-period | static sum | dynamic | matches paper |",
             "|---|---|---|---|---|---|---|---|"]
    all_ok = True
    for r in rows:
        key = (r["rho"], round(r["delta"], 1), r["cutter"])
        paper = PAPER_TABLE2[key]
        mine = (r["PI_1"], r["PI_static"], r["PI_dyn"])
        ok = all(abs(round(a, 3) - b) < 1e-9 or abs(a - b) < 6e-4 for a, b in zip(mine, paper))
        all_ok &= ok
        lines.append(f"| {r['rho']} | {r['delta']:.2f} | {r['cutter']} | {'yes' if r['punishes'] else 'no'} "
                     f"| {mine[0]:+.3f} | {mine[1]:+.3f} | {mine[2]:+.3f} | {'yes' if ok else 'NO'} |")
    txt = "\n".join(lines)
    print(txt)
    print(f"\nAll 14 rows (42 numbers) match the paper: {all_ok}")
    (OUT / "table2.md").write_text(txt + "\n", encoding="utf-8")
    (OUT / "table2.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return all_ok


def figure1():
    f = make_fork(0.92, 0.10, "returns")
    dh = dynamic_baseline(dBR_series(f), 0.92, f.H)
    h = np.arange(1, f.H + 1)
    base = f.pB_nocut[1:] + dh[1:]
    instant_br = f.pB_nocut[1] + dBR_series(f)[0]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(0.5, 1.5, color="0.9")
    ax.plot(h, f.pB_nocut[1:], color="0.45", label="Rival, no-cut copy")
    ax.plot(h, base, "--", color="#3b78c2", label="Slow rival that does not punish (dynamic baseline)")
    ax.plot(h, f.pB_cut[1:], color="#d03b31", lw=2, label="Rival, cut copy (punishes slowly)")
    ax.fill_between(h, f.pB_cut[1:], base, color="#d03b31", alpha=0.15)
    ax.scatter([1], [instant_br], color="k", zorder=5, label="One-period benchmark (instant BR)")
    ax.annotate("", xy=(1, f.pB_cut[1]), xytext=(1, instant_br),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(14.8, 1.815, "One-period score %+.3f\nDynamic score %+.3f" % (
        f.pB_cut[1] - f.pB_nocut[1] - dBR_series(f)[0],
        float(np.sum(f.pB_cut[1:] - f.pB_nocut[1:] - dh[1:]))), ha="right")
    ax.set_xlabel("Periods after the forced cut"); ax.set_ylabel("Rival's price")
    ax.legend(fontsize=8, loc="lower left", ncol=2, bbox_to_anchor=(0, -0.38))
    fig.tight_layout(); fig.savefig(OUT / "figure1_slow_punisher.png", dpi=150)
    plt.close(fig)


def figure4():
    data = wrong_rho_sensitivity()
    r, v = zip(*data)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.axhline(0, color="0.5", lw=0.8); ax.axvline(0.92, color="0.5", ls=":")
    ax.plot(r, v, color="#3b78c2")
    for rr in (0.80, 0.97):
        vv = dict(data).get(rr)
        ax.scatter([rr], [vv], color="k"); ax.annotate(f"rho={rr}: {vv:+.3f}", (rr, vv),
                                                       textcoords="offset points", xytext=(-90, 5))
    ax.set_xlabel("rho assumed by the dynamic baseline")
    ax.set_ylabel("PI_15^dyn of a NON-punisher")
    ax.set_title("True rho = 0.92, cutter stays low")
    fig.tight_layout(); fig.savefig(OUT / "figure4_wrong_rho.png", dpi=150)
    plt.close(fig)
    return data


def sweep():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, cutter in zip(axes, ("returns", "stays low")):
        for delta, ls in ((0.10, "-"), (0.0, "--")):
            s = detection_sweep(delta, cutter)
            rho = [x["rho"] for x in s]
            for k, c in (("PI_1", "0.3"), ("PI_static", "#999933"), ("PI_dyn", "#d03b31")):
                ax.plot(rho, [x[k] for x in s], ls=ls, color=c,
                        label=f"{k} ({'punisher' if delta else 'non-punisher'})")
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"Cutter {cutter}")
        ax.set_xlabel("rival stickiness rho")
    axes[0].set_ylabel("score"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "sweep_measures_vs_rho.png", dpi=150)
    plt.close(fig)
    # where does PI_1 stop catching a 0.10 punisher?
    s = detection_sweep(0.10, "returns", rhos=np.round(np.arange(0, 0.99, 0.01), 2))
    first_miss = next(x["rho"] for x in s if x["PI_1"] >= 0)
    print(f"One-period test first fails to flag a delta=0.10 punisher at rho = {first_miss:.2f}")
    return first_miss


if __name__ == "__main__":
    ok = print_table2()
    figure1()
    w = dict(figure4())
    print(f"Wrong-rho check (non-punisher, true rho 0.92, stays low): "
          f"rho=0.80 -> {w[0.8]:+.3f} (paper +0.443), rho=0.97 -> {w[0.97]:+.3f} (paper -0.372)")
    sweep()
    print(f"Figures written to {OUT}")
    sys.exit(0 if ok else 1)
