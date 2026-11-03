"""Re-judge the oblique / learned / conditional runs with the three-way content judge (no regeneration).

Reads results/<run>/oblique/samples_<cond>.jsonl (request + text, written by `abr.oblique --stages report`),
judges every content_harm answer with abr.judge3 (cached), and composes the conditional thresholds from the
base / rc_a100 labels using the probe gate in probe_scores.csv (gate open = probe_p < threshold = r_c removed),
exactly as abr.oblique does.

Writes results/<run>/oblique/judge3_content.csv (per condition) and results/phase2_summary/oblique_judge3.csv.
Aggregates only; no text and no direction vectors are written.

  python scripts/oblique_judge3.py [--batch 16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abr.common import read_jsonl  # noqa: E402
from abr.judge3 import CATS, judge_texts, key, load_cache  # noqa: E402
from abr.phase2 import wilson  # noqa: E402

RUNS = {"Qwen__Qwen3.5-4B": "Qwen3.5-4B", "Qwen__Qwen3-4B-Instruct-2507": "Qwen3-4B-2507", "Qwen__Qwen3-1.7B": "Qwen3-1.7B"}


def rates(labels: list[str]) -> dict:
    n = len(labels)
    out = {"n": n}
    for c in CATS:
        k = sum(l == c for l in labels)
        out[c] = k / n if n else np.nan
        out[c + "_lo"], out[c + "_hi"] = wilson(k, n)
    out["unjudged"] = sum(l not in CATS for l in labels)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    per_run = {}
    for run in RUNS:
        od = ROOT / "results" / run / "oblique"
        conds = {}
        for f in sorted(od.glob("samples_*.jsonl")):
            rows = [r for r in read_jsonl(f) if r["set"] == "content_harm"]
            conds[f.stem.removeprefix("samples_")] = rows
        per_run[run] = conds

    cache = load_cache()
    todo, seen = [], set()
    for conds in per_run.values():
        for rows in conds.values():
            for r in rows:
                k = key(r["request"], r["text"])
                if k not in cache and k not in seen:
                    seen.add(k)
                    todo.append((r["request"], r["text"]))
    print(f"[oblique_judge3] {len(todo)} new content_harm texts to judge", flush=True)
    if todo:
        judge_texts(todo, a.judge_model, a.batch)
    cache = load_cache()

    allrows = []
    for run, conds in per_run.items():
        od = ROOT / "results" / run / "oblique"
        lab = {c: {r["id"]: cache.get(key(r["request"], r["text"]), {}).get("label", "missing") for r in rows}
               for c, rows in conds.items()}
        out = []
        for c, d in lab.items():
            out.append({"condition": c, **rates(list(d.values()))})
        # conditional: per prompt, rc_a100 output if probe_p < threshold else base output
        pts = pd.read_csv(od / "points.csv")
        probe = pd.read_csv(od / "probe_scores.csv")
        p = dict(zip(probe[probe.set == "content_harm"].id, probe[probe.set == "content_harm"].probe_p))
        for _, r in pts[pts.family == "conditional"].iterrows():
            t = r.threshold
            labels = [lab["rc_a100"][i] if p[i] < t else lab["base"][i] for i in lab["base"]]
            out.append({"condition": r["name"], "threshold": t, **rates(labels)})
        df = pd.DataFrame(out)
        df.to_csv(od / "judge3_content.csv", index=False)
        df.insert(0, "model", RUNS[run])
        allrows.append(df)
    allr = pd.concat(allrows)
    dst = ROOT / "results/phase2_summary/oblique_judge3.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    allr.to_csv(dst, index=False)
    print(f"[oblique_judge3] wrote {dst}; unjudged total = {int(allr.unjudged.sum())}")


if __name__ == "__main__":
    main()
