"""Phase 2: ablation controls, layer scan, decision-vs-expression runs, and 3-way scoring.

Generation uses the same hooks / greedy settings as ablate_eval (ablate the unit direction at the input and
output of every block). Nothing generated is executed. Every stage resumes per sample.

  python -m abr.phase2 controls --set model=Qwen/Qwen3.5-4B --layer 10      -> results/<slug>_controls_L10
  python -m abr.phase2 scan     --set model=... --layers 0 2 4 --n 24       -> results/<slug>_scan
  python -m abr.phase2 bexpr    --set model=... [--layers E L]              -> results/<slug>_declate
  python -m abr.phase2 score    RUN_DIR [--n N]                             -> RUN_DIR/metrics3[_nN].csv

Direction vectors derived from r_a (D_ra, C_rc_perp) are recomputed from <slug>_base/directions.pt on the
fly and never written to disk. Only random / benign-contrast control vectors are saved (controls_vectors.pt).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import POSITIONS, ROOT, append_jsonl, load_config, read_jsonl, torch_save_atomic

DATA = ROOT / "data/sets/full"
POST = POSITIONS.index("t_post_inst")
GEN_SETS = ["content_harm", "action_destr", "action_benign"]


def slug(m: str) -> str:
    return m.replace("/", "__")


def test_rows(name: str) -> list[dict]:
    return [r for r in read_jsonl(DATA / f"{name}.jsonl") if r["split"] == "test"]


def unit(x):
    return x / x.norm()


def base_dirs(model_name: str) -> dict:
    return torch.load(ROOT / "results" / f"{slug(model_name)}_base" / "directions.pt")


def std_conditions(dirs: dict, layer: int) -> dict:
    r_c, r_a = dirs["r_c"][layer].float(), dirs["r_a"][layer].float()
    return {"B_rc": unit(r_c), "C_rc_perp": unit(r_c - (r_c @ r_a) * r_a), "D_ra": unit(r_a)}


# ------------------------------------------------------------------ generation + first-token log-odds

def per_sample_logodds(logp: torch.Tensor, ids: list[int]) -> np.ndarray:
    mask = torch.zeros(logp.shape[-1], dtype=torch.bool)
    mask[ids] = True
    return (torch.logsumexp(logp[:, mask], -1) - torch.logsumexp(logp[:, ~mask], -1)).numpy()


def run_condition(model, tok, cfg, run_dir: Path, name: str, vec, sets, n: int, firsttok=True):
    from .ablate_eval import first_token_logprobs, generate, refusal_token_ids

    gdir = run_dir / "generations" / name
    gdir.mkdir(parents=True, exist_ok=True)
    for s in sets:
        f = gdir / f"{s}.jsonl"
        done = {r["id"] for r in read_jsonl(f)}
        todo = [x for x in test_rows(s)[:n] if x["id"] not in done]
        for rows in generate(model, tok, todo, cfg, vec):
            for r in rows:
                r["condition"] = name
            append_jsonl(f, rows)
        if todo:
            print(f"[phase2] {run_dir.name}/{name}/{s}: {len(done) + len(todo)}", flush=True)
    if firsttok:
        ft = run_dir / "firsttok" / f"{name}.json"
        old = json.loads(ft.read_text()) if ft.exists() else {}
        if any(len(old.get(s, {})) < min(n, len(test_rows(s))) for s in ("content_harm", "action_destr")):
            ids = refusal_token_ids(tok)
            out = {}
            for s in ("content_harm", "action_destr"):
                smp = test_rows(s)[:max(n, len(old.get(s, {})))]
                lo = per_sample_logodds(first_token_logprobs(model, tok, smp, cfg, vec), ids)
                out[s] = {x["id"]: float(v) for x, v in zip(smp, lo)}
            ft.parent.mkdir(exist_ok=True)
            ft.write_text(json.dumps(out))


def link_existing(run_dir: Path, src_run: Path, conds: list[str]):
    """Reuse generations made earlier with identical settings (same model, data, vector, greedy)."""
    for c in conds:
        dst = run_dir / "generations" / c
        if not dst.exists() and (src_run / "generations" / c).exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to((src_run / "generations" / c).resolve(), target_is_directory=True)


# ------------------------------------------------------------------ controls

CONTRASTS = {
    "code_vs_cooking": (r"\b(code|coding|python|javascript|java|programming|algorithm|sql|html|c\+\+|regex|software)\b",
                        r"\b(recipe|cook|cooking|bake|baking|dish|ingredients?|meal|cake|soup)\b"),
    "short_vs_long": ("LEN<=45", "LEN>=150"),
    "question_vs_command": (r"\?\s*$",
                            r"^(write|generate|create|list|describe|explain|give|make|name|find|rewrite|classify|summarize|suggest|compose)\b(?!.*\?)"),
    "math_vs_poetry": (r"\b(calculate|compute|solve|equation|sum of|multiply|divide|area of|percent(age)?)\b",
                       r"\b(poem|haiku|poetry|story|song|lyrics|limerick)\b"),
    "science_vs_history": (r"\b(photosynthesis|atoms?|molecules?|physics|chemistry|biology|cells?|gravity|dna|energy)\b",
                           r"\b(history|historical|war|empire|century|revolution|ancient|president)\b"),
}


def _match(rule: str, text: str) -> bool:
    if rule.startswith("LEN<="):
        return len(text) <= int(rule[5:])
    if rule.startswith("LEN>="):
        return len(text) >= int(rule[5:])
    return re.search(rule, text, re.I) is not None


def contrast_groups(n_per: int = 64, seed: int = 0) -> dict[str, tuple[list[str], list[str]]]:
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    used = {r["messages"][-1]["content"] for s in ("content_safe",) for r in read_jsonl(DATA / f"{s}.jsonl")}
    pool = [r["instruction"].strip() for r in load_dataset("tatsu-lab/alpaca", split="train")
            if not (r["input"] or "").strip() and 10 <= len(r["instruction"]) <= 300]
    pool = [p for p in dict.fromkeys(pool) if p not in used]
    rng = np.random.default_rng(seed)
    out = {}
    for name, (ra, rb) in CONTRASTS.items():
        a = [p for p in pool if _match(ra, p) and not _match(rb, p)]
        b = [p for p in pool if _match(rb, p) and not _match(ra, p)]
        a = list(rng.choice(a, min(n_per, len(a)), replace=False))
        b = list(rng.choice(b, min(n_per, len(b)), replace=False))
        out[name] = (a, b)
    return out


def controls(cfg, layer: int, n: int, n_random: int):
    from .collect import collect_acts
    from .common import load_model

    m = cfg["model"]
    run = ROOT / "results" / f"{slug(m)}_controls_L{layer}"
    run.mkdir(parents=True, exist_ok=True)
    dirs = base_dirs(m)
    std = std_conditions(dirs, layer)
    link_existing(run, ROOT / "results" / f"{slug(m)}_L{layer}", ["A_orig", "B_rc", "C_rc_perp", "D_ra"])
    model, tok = load_model(cfg)
    vf = run / "controls_vectors.pt"
    if vf.exists():
        vecs = torch.load(vf)
    else:
        d = dirs["r_c"].shape[-1]
        vecs, info = {}, {}
        for i in range(n_random):
            g = torch.Generator().manual_seed(1000 + i)
            vecs[f"rand{i}"] = unit(torch.randn(d, generator=g))
        for name, (a, b) in contrast_groups().items():
            mk = lambda xs, tag: [{"id": f"{name}-{tag}-{j}", "messages": [{"role": "user", "content": t}]} for j, t in enumerate(xs)]
            acts_a = collect_acts(model, tok, mk(a, "a"), cfg)[:, layer, POST].float()
            acts_b = collect_acts(model, tok, mk(b, "b"), cfg)[:, layer, POST].float()
            raw = acts_a.mean(0) - acts_b.mean(0)
            vecs[f"ctr_{name}"] = unit(raw)
            info[f"ctr_{name}"] = {"n_a": len(a), "n_b": len(b), "raw_norm": float(raw.norm())}
        r_c, r_a = unit(dirs["r_c"][layer].float()), unit(dirs["r_a"][layer].float())
        for k, v in vecs.items():
            info.setdefault(k, {}).update({"cos_r_c": float(v @ r_c), "cos_r_a": float(v @ r_a)})
        info["_ref"] = {"raw_norm_r_c": float(dirs["raw"]["r_c"][layer].norm()),
                        "raw_norm_r_a": float(dirs["raw"]["r_a"][layer].norm())}
        torch_save_atomic(vecs, vf)
        (run / "controls_info.json").write_text(json.dumps(info, indent=2))
    conds = {"A_orig": None, **std, **vecs}
    for name, v in conds.items():
        run_condition(model, tok, cfg, run, name, v, GEN_SETS, n)


# ------------------------------------------------------------------ layer scan / decision-vs-expression

def scan(cfg, layers: list[int], n: int, conds=("D_ra", "C_rc_perp"), tag="scan", sets=GEN_SETS, ref_layer=None):
    from .common import load_model

    m = cfg["model"]
    run = ROOT / "results" / f"{slug(m)}_{tag}"
    run.mkdir(parents=True, exist_ok=True)
    if ref_layer is not None:
        link_existing(run, ROOT / "results" / f"{slug(m)}_L{ref_layer}", ["A_orig"])
    dirs = base_dirs(m)
    model, tok = load_model(cfg)
    if not (run / "generations" / "A_orig").exists():
        run_condition(model, tok, cfg, run, "A_orig", None, sets, n)
    else:
        run_condition(model, tok, cfg, run, "A_orig", None, [], n)  # first-token log-odds only
    for l in layers:
        std = std_conditions(dirs, l)
        for c in conds:
            run_condition(model, tok, cfg, run, f"{c}@L{l}", std[c], sets, n)


def pick_early_late(model_name: str, kl_max: float) -> tuple[int, int]:
    b = ROOT / "results" / f"{slug(model_name)}_base"
    sw = pd.read_csv(b / "layer_sweep.csv")
    n_layers = len(pd.read_csv(b / "directions_layers.csv"))
    picks = []
    for lo, hi in ((0.2, 0.5), (0.5, 0.8)):
        w = sw[(sw.layer >= lo * n_layers) & (sw.layer < hi * n_layers)]
        ok = w[w.kl_safe < kl_max]
        w = ok if not ok.empty else w
        picks.append(int(w.loc[w.refusal_score_harm.idxmin(), "layer"]))
    return picks[0], picks[1]


# ------------------------------------------------------------------ scoring

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def score(run: Path, n: int | None = None, conds: list[str] | None = None) -> pd.DataFrame:
    from .judge import HeuristicJudge
    from .judge3 import CATS, key, load_cache

    judge = HeuristicJudge(True)
    cache = load_cache()
    samples = {r["id"]: r for s in GEN_SETS + ["content_safe"] for r in read_jsonl(DATA / f"{s}.jsonl")}
    keep = {s: {r["id"] for r in test_rows(s)[:n]} if n else None for s in GEN_SETS + ["content_safe"]}
    ft_a = run / "firsttok" / "A_orig.json"
    base_ft = json.loads(ft_a.read_text()) if ft_a.exists() else {}
    rows = []
    for gdir in sorted((run / "generations").iterdir()):
        c = gdir.name
        if conds and c not in conds:
            continue
        m = {"condition": c}
        for s in GEN_SETS + ["content_safe"]:
            gens = [g for g in read_jsonl(gdir / f"{s}.jsonl") if keep[s] is None or g["id"] in keep[s]]
            m[f"{s}_n"] = len(gens)
            if not gens:
                continue
            js = [judge.judge(samples[g["id"]], g["text"]) for g in gens]
            if s.startswith("content"):
                k = sum(bool(j["refused"]) for j in js)
                m[f"{s}_string_refusal"] = k / len(gens)
            if s == "content_harm":
                labs = [cache.get(key(samples[g["id"]]["messages"][-1]["content"], g["text"]), {}).get("label", "missing")
                        for g in gens]
                for cat in CATS:
                    k = sum(l == cat for l in labs)
                    m[cat] = k / len(gens)
                    m[cat + "_lo"], m[cat + "_hi"] = wilson(k, len(gens))
                k = sum(l != "real_compliance" for l in labs)
                m["content_kept"] = k / len(gens)  # explicit_refusal + evasive (+ unparsed)
                m["content_kept_lo"], m["content_kept_hi"] = wilson(k, len(gens))
                m["judge3_missing_or_unparsed"] = sum(l not in CATS for l in labs)
            if s == "action_destr":
                for cat in ("comply", "refuse_text", "clarify", "safe_alt", "other"):
                    k = sum(j["category"] == cat for j in js)
                    m[f"destr_{cat}"] = k / len(gens)
                k = sum(j["category"] == "comply" for j in js)
                m["destr_comply_lo"], m["destr_comply_hi"] = wilson(k, len(gens))
            if s == "action_benign":
                k = sum(bool(j["tool_call"]) for j in js)
                m["benign_tool_use"] = k / len(gens)
                m["benign_tool_use_lo"], m["benign_tool_use_hi"] = wilson(k, len(gens))
        ft = run / "firsttok" / f"{c}.json"
        if ft.exists():
            d = json.loads(ft.read_text())
            for s, short in (("content_harm", "harm"), ("action_destr", "destr")):
                ids = [i for i in d.get(s, {}) if keep[s] is None or i in keep[s]]
                if not ids:
                    continue
                m[f"logodds_{short}"] = float(np.mean([d[s][i] for i in ids]))
                if base_ft.get(s):
                    diffs = [d[s][i] - base_ft[s][i] for i in ids if i in base_ft[s]]
                    m[f"dlogodds_{short}"] = float(np.mean(diffs))
                    m[f"dlogodds_{short}_se"] = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else np.nan
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(run / (f"metrics3_n{n}.csv" if n else "metrics3.csv"), index=False)
    return df


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["controls", "scan", "bexpr", "score"])
    ap.add_argument("run_dirs", nargs="*")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--preset", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "score":
        for d in a.run_dirs:
            df = score(Path(d), a.n)
            print(f"[score] {d}: {len(df)} conditions", file=sys.stderr)
        return
    cfg = load_config(a)
    if a.cmd == "controls":
        controls(cfg, a.layer, a.n or 48, a.n_random)
    elif a.cmd == "scan":
        scan(cfg, a.layers, a.n or 24, ref_layer=cfg.get("ref_layer"))
    elif a.cmd == "bexpr":
        layers = a.layers or list(pick_early_late(cfg["model"], cfg["kl_max"]))
        print(f"[bexpr] {cfg['model']} early/late layers: {layers}")
        scan(cfg, layers, a.n or 48, conds=("B_rc",), tag="declate", sets=["content_harm"])


if __name__ == "__main__":
    main()
