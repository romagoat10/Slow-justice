"""Part 2: base runs + forked-market audit with Claude Haiku 5.5 (or an offline synthetic
agent with a known rho, for validating the pipeline).

Offline, free, ~seconds (ground truth: sticky rho=0.92 agent with mild punishment):
    python scripts/run_part2.py --backend synthetic --out data/synthetic

Real experiment (costs money; ~6840 Haiku calls, see README for the estimate):
    set ANTHROPIC_API_KEY=...            (or `ant auth login`)
    python scripts/run_part2.py --backend llm --out data/haiku --runs 12

The script is resumable: each finished run is saved to <out>/run_XX.json and skipped on
restart. Positive control (Section 4.2, instructed punisher):
    python scripts/run_part2.py --backend llm --positive-control --out data/haiku_posctrl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctb.agents import LLMAgent, SyntheticAgent
from ctb.game import (FORK_PERIODS, H, Market, base_run, fork_slice, monopoly_check,
                      pick_noise_forks, run_fork, save_run)


def make_agents(args, run_idx: int):
    if args.backend == "llm":
        kw = dict(model=args.model, effort=args.effort, punish_rule=args.positive_control,
                  notes_first=args.notes_first)
        return (LLMAgent(**kw), LLMAgent(**kw))
    seed = 1000 * run_idx
    kw = dict(rho=args.syn_rho, delta=args.syn_delta, noise_sd=args.syn_noise)
    if args.positive_control:
        kw.update(delta=0.4, K=3, rho=0.5, lag=1)   # mimics Haiku acting one period late
    return (SyntheticAgent(seed=seed + 1, **kw), SyntheticAgent(seed=seed + 2, **kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["synthetic", "llm"], default="synthetic")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--T", type=int, default=150)
    ap.add_argument("--forks", type=int, nargs="*", default=list(FORK_PERIODS))
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--notes-first", action="store_true",
                    help="future-work variant: notes before price in the reply format")
    ap.add_argument("--positive-control", action="store_true",
                    help="add the punish rule; 2 games x 100 rounds, forks at 50/60/70/80")
    ap.add_argument("--no-monopoly-check", action="store_true")
    ap.add_argument("--noise-seed", type=int, default=2026)
    ap.add_argument("--syn-rho", type=float, default=0.92)
    ap.add_argument("--syn-delta", type=float, default=0.03)
    ap.add_argument("--syn-noise", type=float, default=0.004)
    args = ap.parse_args()

    if args.positive_control:
        args.runs, args.T, args.forks = min(args.runs, 2), 100, [50, 60, 70, 80]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(
        {k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    # 12 noise/upward points drawn BEFORE any fork is run (paper Section 3.4)
    noise_pts = set() if args.positive_control else set(
        pick_noise_forks(args.runs, tuple(args.forks), 12, args.noise_seed))
    (args.out / "noise_points.json").write_text(json.dumps(sorted(noise_pts)), encoding="utf-8")

    if not args.no_monopoly_check and not (args.out / "monopoly_check.json").exists():
        a, _ = make_agents(args, 999)
        mc = monopoly_check(a, 50)
        print(f"monopoly check: mean price last 10 = {mc['mean_last10']:.3f} (target ~1.802)")
        (args.out / "monopoly_check.json").write_text(json.dumps(mc), encoding="utf-8")

    pool = ThreadPoolExecutor(max_workers=2)
    for r in range(args.runs):
        path = args.out / f"run_{r:02d}.json"
        if path.exists():
            print(f"run {r}: exists, skipping"); continue
        t0 = time.time()
        agents = make_agents(args, r)
        base = base_run(agents, args.T, pool)
        forks = {}
        for k, t in enumerate(args.forks):
            cutter = k % 2                               # alternate cutter between agents
            fk = {"t": t, "cutter": cutter}
            kinds = ["cut", "nocut"]
            if (r, t) in noise_pts:
                kinds += ["nocut2", "raise"]
            for kind in kinds:
                m = run_fork(base, agents, t, cutter, kind, H, pool)
                fk[kind] = fork_slice(m, t, H)
                fk[kind + "_notes"] = [[n.__dict__ for n in row] for row in m.notes[t:t + H + 1]]
                fk[kind + "_rids"] = m.rids[t:t + H + 1]
            forks[str(t)] = fk
        save_run(path, {"run_id": r, "backend": args.backend, "base": base.to_json(),
                        "forks": forks})
        print(f"run {r}: done in {time.time() - t0:.1f}s, "
              f"last-10 mean prices {[round(sum(p[i] for p in base.prices[-10:]) / 10, 3) for i in (0, 1)]}")
    pool.shutdown()


if __name__ == "__main__":
    main()
