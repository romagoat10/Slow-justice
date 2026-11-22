"""What does the rival WRITE after a forced cut? (punishment vs forecasting vs matching)

Reads the raw sprint logs (the agents' plans / insights / observations notes are not copied
into data/haiku because they are large). For each of the 48 main forks, compares the rival's
notes in the cut copy and in the no-cut copy over periods t+1..t+3, and counts which forks
contain language from each category. Keyword matching is crude: treat it as descriptive.

    python scripts/notes_analysis.py --logs <extracted>/logs
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATEGORIES = {
    "punish / retaliate": r"punish|retaliat|price war|tit[- ]for[- ]tat|disciplin|teach (?:firm|them|it)|"
                          r"\bdeter(?:rence|rent|s|red)?\b|\bwarn(?:ing)?\b|send a (?:message|signal)|"
                          r"signal(?:l?ing)? (?:strength|resolve|that i)",
    "match / follow the cut": r"\bmatch(?:ing|ed)?\b|follow(?:ing)? (?:the|its|their|firm)|undercut",
    "forecast: will the cut last?": r"(?:will|may|might|could|whether|if) (?:firm [ab] )?(?:it |they )?"
                                    r"(?:keep|stay|remain|persist|continue|return|revert|go back|hold)",
    "wait and see / monitor": r"monitor|wait|observe whether|see (?:if|whether)|watch",
    "restore cooperation / high prices": r"cooperat|collu|restore|return to (?:the )?(?:previous|high|2\.)|"
                                         r"stabili[sz]e|truce|mutual",
}
COMPILED = {k: re.compile(v, re.I) for k, v in CATEGORIES.items()}
FIRMS = ("Firm A", "Firm B")


def notes_text(actor: dict) -> str:
    r = actor.get("response")
    if isinstance(r, dict):
        return " ".join(str(r.get(k, "")) for k in ("observations", "plans", "insights"))
    return str(r or "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--periods", type=int, default=3)
    args = ap.parse_args()
    logs = args.logs
    census = {l.split("logs/")[-1].strip() for l in (logs / "live_base_census.txt").read_text().splitlines()
              if l.startswith("COMPLETED")}
    copies = defaultdict(dict)          # (base, period) -> kind -> records
    for d in logs.iterdir():
        if not (d / "meta.json").exists() or not d.name.startswith(("fork_cut_", "fork_sham_")):
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        base = meta["base_path"].replace("\\", "/").split("/")[-1]
        if base not in census or meta.get("prompt_variant") is not None:
            continue
        comp = d / "completion.json"
        if not comp.exists() or not json.loads(comp.read_text()).get("complete"):
            continue
        recs = [json.loads(x) for x in open(d / "continuation.jsonl", encoding="utf-8") if x.strip()]
        kind = "cut" if meta["kind"] == "fork_cut" else "nocut"
        copies[(base, meta["fork_period"])].setdefault(kind, (meta["cutter"], recs))

    hits = {c: {"cut": 0, "nocut": 0} for c in CATEGORIES}
    examples = defaultdict(list)
    n = 0
    for key, kinds in sorted(copies.items()):
        if {"cut", "nocut"} - kinds.keys():
            continue
        n += 1
        cutter = kinds["cut"][0]
        rival = [f for f in FIRMS if f != cutter][0]
        for kind in ("cut", "nocut"):
            recs = kinds[kind][1]
            text = " ".join(notes_text(r["actors"][rival]) for r in recs[1:1 + args.periods])
            for c, rx in COMPILED.items():
                m = rx.search(text)
                if m:
                    hits[c][kind] += 1
                    if kind == "cut" and len(examples[c]) < 2:
                        s = max(0, m.start() - 90)
                        examples[c].append(text[s:m.end() + 90].replace("\n", " "))
    print(f"Rival's notes, periods t+1..t+{args.periods}, {n} forks (share of forks with that language)\n")
    print(f"{'category':34s} {'cut copy':>10s} {'no-cut copy':>12s}")
    for c in CATEGORIES:
        print(f"{c:34s} {hits[c]['cut'] / n:10.0%} {hits[c]['nocut'] / n:12.0%}")
    print("\nExamples from cut copies:")
    for c, ex in examples.items():
        for e in ex:
            print(f"  [{c}] ...{e}...")


if __name__ == "__main__":
    main()
