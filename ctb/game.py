"""Repeated duopoly game, forked-market audit, and log format (Sections 3.4 and 4.2).

Timeline of one fork at period t with cutter i (rival j = 1 - i):

    periods 0 .. t-1   copied from the base run (prices AND both agents' notes)
    period  t          rival j: its base-run price p_j(t) (it moved before seeing anything)
                       cutter i: cut copy    -> round(0.9 * p_i(t-1), 2)    (forced, unseen)
                                 raise copy  -> round(1.1 * p_i(t-1), 2)    (upward control)
                                 no-cut copy -> p_i(t) from the base run
                       both agents' period-t notes are the base-run notes; the cutter's
                       history shows the forced price as if it were its own.
    periods t+1..t+H   both agents are called fresh in every copy (H = 15 -> 30 calls/copy)

Call budget check against the paper (Section 4.2): 12 base runs x 150 periods x 2 agents
= 3600; 48 forks x 2 copies x 30 = 2880; 12 second no-cut copies x 30 = 360;
total 6840, exactly the "6840 model calls" the paper reports.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .agents import Agent, Decision, Notes, Period
from .market import demand, monopoly_demand, monopoly_profit, profit

FORK_PERIODS = (75, 95, 115, 135)
H = 15
CUT = 0.90
RAISE = 1.10


class Market:
    """Mutable state of one copy of the market."""

    def __init__(self):
        self.prices: list[list[float]] = []     # [t] -> [p0, p1]
        self.notes: list[list[Notes]] = []      # [t] -> [notes0, notes1]
        self.rids: list[list[str]] = []

    def copy_until(self, t: int) -> "Market":
        m = Market()
        m.prices = [list(p) for p in self.prices[:t]]
        m.notes = [list(n) for n in self.notes[:t]]
        m.rids = [list(r) for r in self.rids[:t]]
        return m

    def history_for(self, i: int) -> list[Period]:
        out = []
        for t, (p) in enumerate(self.prices):
            mine, rival = p[i], p[1 - i]
            out.append(Period(t, mine, rival, demand(mine, rival), profit(mine, rival)))
        return out

    def last_notes(self, i: int) -> Notes:
        return self.notes[-1][i] if self.notes else Notes()

    def step(self, agents: tuple[Agent, Agent], pool: ThreadPoolExecutor | None = None):
        hists = [self.history_for(i) for i in (0, 1)]
        notes = [self.last_notes(i) for i in (0, 1)]
        if pool is None:
            ds = [agents[i].act(hists[i], notes[i]) for i in (0, 1)]
        else:
            futs = [pool.submit(agents[i].act, hists[i], notes[i]) for i in (0, 1)]
            ds = [f.result() for f in futs]
        self.prices.append([ds[0].price, ds[1].price])
        self.notes.append([ds[0].notes, ds[1].notes])
        self.rids.append([ds[0].response_id, ds[1].response_id])
        return ds

    def to_json(self) -> dict:
        return {"prices": self.prices,
                "notes": [[asdict(n) for n in row] for row in self.notes],
                "response_ids": self.rids}

    @staticmethod
    def from_json(d: dict) -> "Market":
        m = Market()
        m.prices = d["prices"]
        m.notes = [[Notes(**n) for n in row] for row in d["notes"]]
        m.rids = d.get("response_ids", [["", ""] for _ in m.prices])
        return m


def monopoly_check(agent: Agent, periods: int = 50) -> dict:
    """Fish et al. qualification: one agent alone; mean price over last 10 periods."""
    hist: list[Period] = []
    notes = Notes()
    prices = []
    for t in range(periods):
        d = agent.act(hist, notes, monopoly=True)
        notes = d.notes
        prices.append(d.price)
        hist.append(Period(t, d.price, float("nan"), monopoly_demand(d.price),
                           monopoly_profit(d.price)))
    return {"prices": prices, "mean_last10": float(np.mean(prices[-10:]))}


def base_run(agents, T: int = 150, pool=None, progress: Callable | None = None) -> Market:
    m = Market()
    for t in range(T):
        m.step(agents, pool)
        if progress:
            progress(t)
    return m


def run_fork(base: Market, agents, t: int, cutter: int, kind: str, H: int = H,
             pool=None) -> Market:
    """kind in {'cut', 'nocut', 'nocut2', 'raise'}."""
    m = base.copy_until(t)
    p_base_t = list(base.prices[t])
    prev = base.prices[t - 1][cutter]
    if kind == "cut":
        p_base_t[cutter] = round(CUT * prev, 2)
    elif kind == "raise":
        p_base_t[cutter] = round(RAISE * prev, 2)
    elif kind not in ("nocut", "nocut2"):
        raise ValueError(kind)
    m.prices.append(p_base_t)
    m.notes.append(list(base.notes[t]))
    m.rids.append(list(base.rids[t]))
    for _ in range(H):
        m.step(agents, pool)
    return m


def fork_slice(m: Market, t: int, H: int = H) -> list[list[float]]:
    return m.prices[t:t + H + 1]


def pick_noise_forks(n_runs: int, fork_periods=FORK_PERIODS, k: int = 12, seed: int = 2026):
    """12 (run, fork) points drawn with a fixed seed BEFORE any fork is run."""
    rng = np.random.default_rng(seed)
    all_pts = [(r, t) for r in range(n_runs) for t in fork_periods]
    idx = rng.choice(len(all_pts), size=min(k, len(all_pts)), replace=False)
    return sorted(all_pts[i] for i in idx)


def save_run(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def load_runs(folder: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(folder).glob("run_*.json"))]
