"""Part 2 analysis: Table 3, rho grid, per-run rho, noise floor, upward control,
deterrence, Figure 2 data, and positive-control table (Table 5).

Input: a list of run dicts as written by scripts/run_part2.py:

    {
      "run_id": 0,
      "base":  Market.to_json()            # 150 periods
      "forks": {
          "75":  {"t": 75, "cutter": 0,
                  "cut":    [[pA, pB], ... H+1 rows],
                  "nocut":  [...],
                  "nocut2": [...]   (only on the 12 noise points)
                  "raise":  [...]   (only on the 12 noise points)
                 }, ...
      }
    }
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measures import (Fork, deterrence, deterrence_enforced, dBR_series, d_series,
                       dynamic_baseline, pi_one_period, pi_static)
from .stats import bootstrap_runs, sign_test_p
from .stickiness import fixed_effects_rho, ols, per_run_rho, run_rows

H = 15
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)


def make_fork(fk: dict, cut_key: str = "cut", base_key: str = "nocut") -> Fork:
    i = fk["cutter"]
    j = 1 - i
    c = np.asarray(fk[cut_key], float)
    n = np.asarray(fk[base_key], float)
    return Fork(pA_cut=c[:, i], pA_nocut=n[:, i], pB_cut=c[:, j], pB_nocut=n[:, j],
                meta=dict(t=fk["t"], cutter=i))


def _shown(run: dict) -> dict:
    return {"shown": np.asarray(run["base"]["prices"], float)}


@dataclass
class ForkCache:
    run: int
    t: int
    d: np.ndarray
    dBR: np.ndarray
    f: Fork

    def dyn(self, rho: float) -> float:
        dh = dynamic_baseline(self.dBR, rho, H)
        return float(np.sum(self.d[1:H + 1] - dh[1:H + 1]))


def _cache(runs, cut_key="cut", base_key="nocut", only_with=None):
    out = []
    for r, run in enumerate(runs):
        for key, fk in sorted(run["forks"].items(), key=lambda kv: int(kv[0])):
            if cut_key not in fk or base_key not in fk:
                continue
            if only_with and only_with not in fk:
                continue
            f = make_fork(fk, cut_key, base_key)
            out.append(ForkCache(r, fk["t"], d_series(f), dBR_series(f), f))
    return out


def _run_means(values: np.ndarray, run_ids: np.ndarray, n_runs: int) -> np.ndarray:
    return np.array([values[run_ids == r].mean() for r in range(n_runs)])


def table3(runs: list[dict], B: int = 10_000, seed: int = 0) -> dict:
    n = len(runs)
    rows = [run_rows(_shown(r)) for r in runs]                  # per run: [(x,y) agent0, agent1]
    X = [np.concatenate([a[0] for a in rr]) for rr in rows]
    Y = [np.concatenate([a[1] for a in rr]) for rr in rows]

    def rho_of(idx):
        return ols(np.concatenate([X[i] for i in idx]), np.concatenate([Y[i] for i in idx]))[1]

    rho_hat = rho_of(np.arange(n))
    forks = _cache(runs)
    rid = np.array([fc.run for fc in forks])

    pi1 = np.array([pi_one_period(fc.f) for fc in forks])
    pst = np.array([pi_static(fc.f, H) for fc in forks])
    G = np.array([deterrence(fc.f, H) for fc in forks])

    def dyn_at(rho):
        return np.array([fc.dyn(rho) for fc in forks])

    def genf_at(rho):
        return np.array([deterrence_enforced(fc.f, rho, H) for fc in forks])

    pdyn, genf = dyn_at(rho_hat), genf_at(rho_hat)
    per_run = {k: _run_means(v, rid, n) for k, v in
               dict(PI_1=pi1, PI_static=pst, PI_dyn=pdyn, G=G, G_enforced=genf).items()}

    res = {"rho_hat": rho_hat, "n_runs": n, "n_forks": len(forks), "measures": {}}
    # Bootstrap: resample runs; rho re-estimated in every resample for rho-dependent measures.
    for k in per_run:
        vals = per_run[k]
        if k in ("PI_dyn", "G_enforced"):
            per_fork_fn = dyn_at if k == "PI_dyn" else genf_at
            memo = {}

            def stat(idx, per_fork_fn=per_fork_fn, memo=memo):
                rho = round(rho_of(idx), 4)                  # memoize on rounded rho
                if rho not in memo:
                    memo[rho] = _run_means(per_fork_fn(rho), rid, n)
                return float(memo[rho][idx].mean())
        else:
            def stat(idx, vals=vals):
                return float(vals[idx].mean())
        lo, hi, _ = bootstrap_runs(n, stat, B=B, seed=seed)
        below, nz, p = sign_test_p(vals)
        res["measures"][k] = dict(mean=float(vals.mean()), ci=(lo, hi), runs_below=below,
                                  n=nz, sign_p=p, per_run=vals.tolist())

    lo, hi, _ = bootstrap_runs(n, lambda idx: rho_of(idx), B=B, seed=seed)
    res["rho_ci"] = (lo, hi)
    res["per_fork"] = dict(PI_1=pi1.tolist(), PI_static=pst.tolist(), PI_dyn=pdyn.tolist(),
                           G=G.tolist(), run=rid.tolist(), t=[fc.t for fc in forks])
    res["disagree"] = int(np.sum((pi1 >= 0) & (pdyn < 0)))
    res["cut_paid_off"] = int(np.sum(G > 0))

    # rho grid (fixed rho, intervals from resampling runs only)
    grid = {}
    for rho in RHO_GRID:
        v = _run_means(dyn_at(rho), rid, n)
        lo, hi, _ = bootstrap_runs(n, lambda idx, v=v: float(v[idx].mean()), B=B, seed=seed)
        grid[rho] = dict(mean=float(v.mean()), ci=(lo, hi))
    res["rho_grid"] = grid

    # each run's own rho
    own = np.array([per_run_rho(_shown(r)) for r in runs])
    v_own = np.array([np.mean([fc.dyn(own[fc.run]) for fc in forks if fc.run == r])
                      for r in range(n)])
    lo, hi, _ = bootstrap_runs(n, lambda idx: float(v_own[idx].mean()), B=B, seed=seed)
    res["own_rho"] = dict(per_run_rho=own.tolist(), mean=float(v_own.mean()), ci=(lo, hi))

    # extension: fixed-effects rho
    res["rho_fixed_effects"] = fixed_effects_rho([_shown(r) for r in runs])

    res["noise"] = noise_floor(runs, rho_hat)
    res["upward"] = upward_control(runs, rho_hat, B=B, seed=seed)
    res["last10_mean_price"] = float(np.mean(
        [np.mean(np.asarray(r["base"]["prices"])[-10:]) for r in runs]))
    return res


def noise_floor(runs, rho) -> dict:
    """Placebo: score the second no-cut copy as if it were the cut copy.

    Paper reports '95th pct. of absolute no-cut vs second no-cut gap' = 0.007. The PDF does
    not say whether 'gap' is per agent-period price or per fork score; we report both.
    """
    fcs = _cache(runs, cut_key="nocut2", base_key="nocut")
    if not fcs:
        return {}
    scores = np.array([abs(fc.dyn(rho)) for fc in fcs])
    same, total, gaps = 0, 0, []
    for run in runs:
        for fk in run["forks"].values():
            if "nocut2" in fk:
                a = np.asarray(fk["nocut"])[1:]
                b = np.asarray(fk["nocut2"])[1:]
                same += int(np.sum(np.isclose(a, b)))
                total += a.size
                gaps.extend(np.abs(a - b).ravel().tolist())
    return dict(n_points=len(fcs), p95_abs_dyn_score=float(np.quantile(scores, 0.95)),
                p95_abs_price_gap=float(np.quantile(gaps, 0.95)),
                identical_agent_periods=same, agent_periods=total)


def upward_control(runs, rho, B=10_000, seed=0) -> dict:
    up = _cache(runs, cut_key="raise", base_key="nocut")
    if not up:
        return {}
    keys = {(fc.run, fc.t) for fc in up}
    cut_same = [fc for fc in _cache(runs) if (fc.run, fc.t) in keys]
    runs_used = sorted({fc.run for fc in up})
    def run_mean(fcs, fn):
        return np.array([np.mean([fn(fc) for fc in fcs if fc.run == r]) for r in runs_used])
    v_up = run_mean(up, lambda fc: fc.dyn(rho))
    v_cut = run_mean(cut_same, lambda fc: fc.dyn(rho))
    lo, hi, _ = bootstrap_runs(len(v_up), lambda idx: float(v_up[idx].mean()), B=B, seed=seed)
    return dict(n_forks=len(up), n_runs=len(runs_used),
                dyn_after_raise=float(v_up.mean()), ci=(lo, hi),
                runs_negative_after_raise=int(np.sum(v_up < 0)),
                dyn_after_cut_same_forks=float(v_cut.mean()),
                pi1_after_raise=float(np.mean([pi_one_period(fc.f) for fc in up])))


def positive_control_table(game: dict, rho: float | None = None) -> dict:
    """Table 5 for one positive-control game. rho defaults to the game's own estimate."""
    if rho is None:
        rho = per_run_rho(_shown(game), t_min=1, t_max=min(49, len(game["base"]["prices"]) - 1))
    rows = []
    for key, fk in sorted(game["forks"].items(), key=lambda kv: int(kv[0])):
        f = make_fork(fk)
        fc = ForkCache(0, fk["t"], d_series(f), dBR_series(f), f)
        rows.append(dict(t=fk["t"], PI_1=pi_one_period(f), PI_static=pi_static(f, H),
                         PI_dyn=fc.dyn(rho)))
    return dict(rho=rho, rows=rows)


def format_table3(res: dict) -> str:
    names = [("PI_1", "One-period PI_1"), ("PI_static", "Static sum PI_15^static"),
             ("PI_dyn", f"Dynamic PI_15^dyn (rho={res['rho_hat']:.3f})"),
             ("G", "G_15 (cutter gain)"), ("G_enforced", "G_15^enforced")]
    out = [f"{'Measure':38s} {'Mean':>8s}  {'95% CI':>20s}  {'Runs<0':>7s}  {'sign p':>7s}"]
    for k, label in names:
        m = res["measures"][k]
        out.append(f"{label:38s} {m['mean']:+8.3f}  [{m['ci'][0]:+.3f}, {m['ci'][1]:+.3f}]"
                   f"  {m['runs_below']:>3d}/{m['n']:<3d}  {m['sign_p']:7.4f}")
    nf = res.get("noise") or {}
    if nf:
        out.append(f"{'Noise floor (p95 |placebo PI_dyn|)':38s} {nf['p95_abs_dyn_score']:8.3f}")
    return "\n".join(out)
