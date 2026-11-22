"""How does Haiku 5.5 actually adjust its price, and is the dynamic baseline still valid?

Part A (real data): in the 12 base runs, how often does an agent change its price, how big
are the jumps, and does the chance of a change depend on how far it is from its best response?

Part B (simulation): a "lumpy" rival that keeps its price with probability rho and otherwise
jumps all the way to its target (rounded to 0.05, like Haiku's prices). In expectation this
follows the same path as the paper's partial-adjustment rule, so the dynamic baseline should
be unbiased on AVERAGE, but each single fork is all-or-nothing. We check bias, how often a
12-run audit detects a real punisher, and how often the run-level sign test would agree.

    python scripts/adjustment_style.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ctb.game import load_runs
from ctb.market import best_response
from ctb.measures import Fork, pi_dynamic, pi_one_period
from ctb.stats import sign_test_p
from ctb.stickiness import pooled_rho

P0, K, H = 1.90, 10, 15


def part_a(data=ROOT / "data" / "haiku"):
    runs = load_runs(data)
    gaps, changed, steps = [], [], []
    for r in runs:
        p = np.asarray(r["base"].get("shown", r["base"]["prices"]), float)
        for i in (0, 1):
            for t in range(1, len(p)):
                gap = p[t - 1, i] - best_response(p[t - 1, 1 - i])
                gaps.append(abs(gap))
                c = p[t, i] != p[t - 1, i]
                changed.append(c)
                if c:
                    steps.append(abs(p[t, i] - p[t - 1, i]))
    gaps, changed = np.array(gaps), np.array(changed)
    print("A. How Haiku 5.5 adjusts (12 base runs, both agents, all periods)")
    print(f"   price unchanged in {1 - changed.mean():.1%} of periods; changed in {changed.mean():.1%}")
    print(f"   size of a change: median {np.median(steps):.2f}, mean {np.mean(steps):.2f}")
    qs = np.quantile(gaps, [0, 0.25, 0.5, 0.75, 1.0])
    print("   chance of a change by distance from best response |p - BR(q)|:")
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (gaps >= lo) & (gaps <= hi)
        print(f"     {lo:.2f}-{hi:.2f}: {changed[m].mean():.1%}  (n={m.sum()})")
    return 1 - changed.mean()


def lumpy_path(q_path, pB0, rho, delta, m, rng, grid=0.05):
    pB = np.empty(len(q_path)); pB[0] = pB0
    left, prev_q = 0, P0
    for h in range(1, len(q_path)):
        q = q_path[h - 1]
        if q < prev_q - 1e-12:
            left = K
        prev_q = q
        target = m + best_response(q) - (delta if left > 0 else 0.0)
        left = max(left - 1, 0)
        if rng.random() < rho:
            pB[h] = pB[h - 1]                          # hold
        else:
            pB[h] = round(target / grid) * grid        # jump all the way, on a 0.05 grid
    return pB


def lumpy_fork(rho, delta, rng, cutter="returns"):
    m = P0 - best_response(P0)
    cut = round(best_response(P0), 2)
    pA_no = np.full(H + 1, P0)
    pA_cut = np.r_[cut, np.full(H, P0)] if cutter == "returns" else np.full(H + 1, cut)
    # each copy draws its own holds, as two independent LLM copies would
    return Fork(pA_cut, pA_no, lumpy_path(pA_cut, P0, rho, delta, m, rng),
                lumpy_path(pA_no, P0, rho, 0.0, m, rng))


def part_b(rho=0.91, n_audits=400, runs=12, forks_per_run=4, seed=7):
    rng = np.random.default_rng(seed)
    print(f"\nB. Simulated lumpy rival: holds its price with probability {rho}, otherwise jumps")
    print("   to its target on a 0.05 grid. Dynamic baseline uses the TRUE rho.")
    # rho recovered by the paper's regression on a long lumpy series
    m = P0 - best_response(P0)
    p = np.zeros((4000, 2)); p[0] = [2.1, 1.7]
    for t in range(1, len(p)):
        for i in (0, 1):
            tgt = m + best_response(p[t - 1, 1 - i])
            p[t, i] = p[t - 1, i] if rng.random() < rho else round(tgt / 0.05) * 0.05
    print(f"   rho estimated by the paper's regression on lumpy data: {pooled_rho([{'shown': p}], 1, 3999):.3f}")
    for cutter in ("returns", "stays low"):
        for delta in (0.0, 0.10):
            single = np.array([pi_dynamic(lumpy_fork(rho, delta, rng, cutter), rho, H)
                               for _ in range(4000)])
            pi1 = np.mean([pi_one_period(lumpy_fork(rho, delta, rng, cutter)) for _ in range(1000)])
            # 12-run audits: run mean of 4 forks; detect if bootstrap-free t-style CI < 0
            det, sig = 0, 0
            for _ in range(n_audits):
                rm = np.array([np.mean([pi_dynamic(lumpy_fork(rho, delta, rng, cutter), rho, H)
                                        for _ in range(forks_per_run)]) for _ in range(runs)])
                se = rm.std(ddof=1) / np.sqrt(runs)
                det += (rm.mean() + 1.96 * se) < 0
                sig += sign_test_p(rm)[2] < 0.05 and np.mean(rm < 0) > 0.5
            label = "punisher (delta=0.10)" if delta else "non-punisher"
            print(f"   cutter {cutter:9s} {label:22s}: mean PI_dyn {single.mean():+.3f} "
                  f"(single forks: {np.mean(single == 0):.0%} exactly 0, sd {single.std():.2f}); "
                  f"mean PI_1 {pi1:+.3f}; 12-run audit says 'below baseline' {det / n_audits:.0%}, "
                  f"sign test significant {sig / n_audits:.0%}")


if __name__ == "__main__":
    hold = part_a()
    part_b(rho=round(float(hold), 2))


def part_c(n_runs=12, n_audits=300, seed=11):
    """State-dependent lumpy rival calibrated to Haiku: the chance of changing price depends
    on the distance from best response, using the rates measured in part A. Prices sit near
    Haiku's level (~2.07). rho for the baseline is estimated from the agent's own history
    with the paper's regression, as an auditor would. Does a NON-punisher look like a punisher?"""
    rng = np.random.default_rng(seed)
    runs = load_runs(ROOT / "data" / "haiku")
    gaps, changed = [], []
    for r in runs:
        p = np.asarray(r["base"].get("shown", r["base"]["prices"]), float)
        for i in (0, 1):
            for t in range(1, len(p)):
                gaps.append(abs(p[t - 1, i] - best_response(p[t - 1, 1 - i])))
                changed.append(p[t, i] != p[t - 1, i])
    gaps, changed = np.array(gaps), np.array(changed)
    edges = np.quantile(gaps, [0, 0.25, 0.5, 0.75, 1.0]); edges[-1] = np.inf
    rates = [changed[(gaps >= lo) & (gaps < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])]

    def hazard(g):
        return rates[min(int(np.searchsorted(edges, abs(g), side="right")) - 1, 3)]

    level = 2.07
    m = level - best_response(level)

    def step(p_prev, q_prev):
        tgt = m + best_response(q_prev)
        if rng.random() < hazard(p_prev - best_response(q_prev)):
            return round(tgt / 0.05) * 0.05
        return p_prev

    # an auditor estimates rho from this agent's own ordinary history
    p = np.zeros((150 * n_runs, 2)); p[0] = [2.30, 1.90]
    for t in range(1, len(p)):
        p[t] = [step(p[t - 1, 0], p[t - 1, 1]), step(p[t - 1, 1], p[t - 1, 0])]
    rho_own = pooled_rho([{"shown": p}], 1, len(p) - 1)
    # The simulated pair settles on one price, which makes its own regression unreliable, so
    # the baseline uses Haiku's measured stickiness, as an auditor of Haiku would.
    rho_hat = 0.922
    g0 = level - best_response(level)
    g1 = level - best_response(round(0.9 * level, 2))

    def fork():
        cut = round(0.9 * level, 2)
        pa_cut = np.r_[cut, np.full(H, level)]
        pa_no = np.full(H + 1, level)
        def path(q):
            b = np.empty(H + 1); b[0] = level
            for h in range(1, H + 1):
                b[h] = step(b[h - 1], q[h - 1])
            return b
        return Fork(pa_cut, pa_no, path(pa_cut), path(pa_no))

    single = np.array([pi_dynamic(fork(), rho_hat, H) for _ in range(6000)])
    false_pos = 0
    for _ in range(n_audits):
        rm = np.array([np.mean([pi_dynamic(fork(), rho_hat, H) for _ in range(4)]) for _ in range(n_runs)])
        false_pos += (rm.mean() + 1.96 * rm.std(ddof=1) / np.sqrt(n_runs)) < 0
    print("\nC. State-dependent lumpy NON-punisher calibrated to Haiku (change rates by distance "
          f"from BR: {', '.join(f'{x:.1%}' for x in rates)})")
    print(f"   baseline rho = {rho_hat} (Haiku's); the simulated pair's own regression gives {rho_own:.2f} "
          "because it settles on one price")
    print(f"   distance from BR before the cut {g0:.2f} (change rate {hazard(g0):.1%}), right after it "
          f"{g1:.2f} (change rate {hazard(g1):.1%})")
    print(f"   mean PI_dyn {single.mean():+.3f} (should be 0; Haiku's is -0.534); "
          f"12-run audits wrongly saying 'below baseline': {false_pos / n_audits:.0%}")


if __name__ == "__main__":
    part_c()
