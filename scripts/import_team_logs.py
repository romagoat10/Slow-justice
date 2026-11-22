"""Convert the original hackathon Haiku 5.5 logs (results/live_logs.tar.gz in the sprint repo)
into this project's run format, so scripts/analyze_part2.py can reanalyze them for free.

    python scripts/import_team_logs.py --logs <extracted>/logs --out data/haiku

Mapping (sprint log -> this project):
    base_<id>/rounds.jsonl          one record per period (period 1..150)  -> run["base"]
    fork_cut_<id>/continuation.jsonl periods t..t+15 (16 records)           -> fork["cut"]
    fork_sham_<id>                  untouched copy                          -> fork["nocut"]
    fork_sham2_<id>                 second untouched copy                   -> fork["nocut2"]
    fork_up_<id>                    forced +10% raise                       -> fork["raise"]

Only base runs listed as COMPLETED in live_base_census.txt are used for the main audit.
Base runs with prompt_variant "punisher_v2" that have forks are the positive-control games used in
the paper (third attempt; --posctrl-variant punisher selects the failed earlier attempt).
Fork "t" is stored as an index into run["base"]["prices"] (period p -> index p-1).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctb.game import save_run

FIRMS = ("Firm A", "Firm B")
KIND = {"fork_cut": "cut", "fork_sham": "nocut", "fork_sham2": "nocut2", "fork_up": "raise"}


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prices(records: list[dict]) -> list[list[float]]:
    return [[float(r["actors"][a]["executed_price"]) for a in FIRMS] for r in records]


def load(logs: Path):
    census = set()
    cf = logs / "live_base_census.txt"
    if cf.exists():
        census = {line.split("logs/")[-1].strip() for line in cf.read_text().splitlines()
                  if line.startswith("COMPLETED")}
    bases, forks = {}, defaultdict(list)
    for d in sorted(logs.iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if d.name.startswith("base_"):
            bases[d.name] = (meta, d)
        elif meta.get("kind") in KIND:
            comp = d / "completion.json"
            ok = comp.exists() and json.loads(comp.read_text()).get("complete")
            forks[meta["base_path"].replace("\\", "/").split("/")[-1]].append((meta, d, ok))
    return census, bases, forks


def build_run(run_id, base_name, meta, d, fork_list, H=15):
    recs = read_jsonl(d / "rounds.jsonl")
    periods = [r["period"] for r in recs]
    assert periods == list(range(1, len(recs) + 1)), f"{base_name}: periods not 1..N"
    run = {"run_id": run_id, "source": base_name, "backend": "imported-sprint-logs",
           "prompt_variant": meta.get("prompt_variant"),
           "base": {"prices": prices(recs)}, "forks": {}}
    issues = []
    for fmeta, fd, ok in fork_list:
        kind = KIND[fmeta["kind"]]
        if not ok:
            issues.append(f"skip incomplete {fd.name}"); continue
        cont = read_jsonl(fd / "continuation.jsonl")
        t = int(fmeta["fork_period"])
        if [c["period"] for c in cont] != list(range(t, t + H + 1)):
            issues.append(f"skip {fd.name}: periods {cont[0]['period']}..{cont[-1]['period']}"); continue
        key = str(t - 1)
        fk = run["forks"].setdefault(key, {"t": t - 1, "period": t,
                                           "cutter": FIRMS.index(fmeta["cutter"])})
        if fk["cutter"] != FIRMS.index(fmeta["cutter"]):
            issues.append(f"cutter mismatch at {t} in {fd.name}"); continue
        if kind in fk:
            issues.append(f"duplicate {kind} at period {t} ({fd.name}); keeping first"); continue
        fk[kind] = prices(cont)
        fk[kind + "_dir"] = fd.name
    # forks without a cut/nocut pair are unusable for the main measures
    for key in list(run["forks"]):
        if not {"cut", "nocut"} <= run["forks"][key].keys():
            issues.append(f"period {run['forks'][key]['period']}: no cut/nocut pair, dropped")
            del run["forks"][key]
    return run, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "haiku")
    ap.add_argument("--posctrl-out", type=Path, default=ROOT / "data" / "haiku_posctrl")
    ap.add_argument("--posctrl-variant", default="punisher_v2")
    args = ap.parse_args()
    census, bases, forks = load(args.logs)

    main_runs = [b for b in sorted(bases) if b in census and bases[b][0].get("prompt_variant") is None]
    pos_runs = [b for b in sorted(bases) if bases[b][0].get("prompt_variant") == args.posctrl_variant
                and forks.get(b)]
    print(f"{len(bases)} base runs found; {len(main_runs)} completed main runs; "
          f"{len(pos_runs)} positive-control ('{args.posctrl_variant}') games")
    for out, names in ((args.out, main_runs), (args.posctrl_out, pos_runs)):
        for i, name in enumerate(names):
            meta, d = bases[name]
            run, issues = build_run(i, name, meta, d, forks.get(name, []))
            save_run(out / f"run_{i:02d}.json", run)
            kinds = defaultdict(int)
            for fk in run["forks"].values():
                for k in ("cut", "nocut", "nocut2", "raise"):
                    kinds[k] += k in fk
            print(f"  {out.name}/run_{i:02d} <- {name}: {len(run['base']['prices'])} periods, "
                  f"forks at periods {sorted(f['period'] for f in run['forks'].values())}, {dict(kinds)}")
            for msg in issues:
                print(f"      note: {msg}")


if __name__ == "__main__":
    main()
