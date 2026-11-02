"""Diff-of-means directions per layer from TRAIN activations (CPU only; needs only activations/).

  r_c = mean(content_harm) - mean(content_safe)   at t_post_inst   (content refusal)
  r_a = mean(action_destr) - mean(action_benign)  at t_post_inst   (action caution)
  h_c, h_a: same contrasts at t_inst                                 (harmfulness)

Also: split-half reliability baselines (cos between directions from two random halves of the same
sets), and held-out (test) AUROC of each direction's projection on both the content and action task.
Writes directions.pt and directions_layers.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .common import POSITIONS, load_config, paths, parse_args, read_jsonl, torch_save_atomic

INST, POST = POSITIONS.index("t_inst"), POSITIONS.index("t_post_inst")

# direction name -> (positive set, negative set, position index)
SPECS = {
    "r_c": ("content_harm", "content_safe", POST),
    "r_a": ("action_destr", "action_benign", POST),
    "h_c": ("content_harm", "content_safe", INST),
    "h_a": ("action_destr", "action_benign", INST),
}
TASKS = {"content": ("content_harm", "content_safe"), "action": ("action_destr", "action_benign")}


def load_acts(p, cfg, name: str, split: str) -> torch.Tensor:
    d = torch.load(p.acts / f"{name}_{split}.pt")
    acts = d["acts"].float()
    if name == "action_destr" and split == "train" and cfg.get("exclude_ambiguous_from_directions", True):
        amb = {r["id"] for r in read_jsonl(p.set_file(name)) if r.get("ambiguous")}
        keep = [i for i, x in enumerate(d["ids"]) if x not in amb]
        acts = acts[keep]
    return acts  # [N, L, 2, d]


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (unit(a) * unit(b)).sum(-1)


def diff_means(pos: torch.Tensor, neg: torch.Tensor, k: int) -> torch.Tensor:
    return pos[:, :, k].mean(0) - neg[:, :, k].mean(0)  # [L, d]


def split_half(pos, neg, k, n_rep, rng, paired: bool):
    """cos between diff-of-means directions from disjoint random halves; [n_rep, L]."""
    out = []
    for _ in range(n_rep):
        ip = rng.permutation(len(pos))
        ineg = ip if paired and len(pos) == len(neg) else rng.permutation(len(neg))
        hp, hn = len(ip) // 2, len(ineg) // 2
        d1 = diff_means(pos[ip[:hp]], neg[ineg[:hn]], k)
        d2 = diff_means(pos[ip[hp:]], neg[ineg[hn:]], k)
        out.append(cos(d1, d2))
    return torch.stack(out)


def compute(cfg) -> dict:
    p = paths(cfg)
    rng = np.random.default_rng(cfg["seed"])
    train = {s: load_acts(p, cfg, s, "train") for s in ("content_harm", "content_safe", "action_destr", "action_benign")}
    test = {s: load_acts(p, cfg, s, "test") for s in train}
    n_layers = train["content_harm"].shape[1]

    raw = {k: diff_means(train[a], train[b], pos) for k, (a, b, pos) in SPECS.items()}
    dirs = {k: unit(v) for k, v in raw.items()}

    df = pd.DataFrame({"layer": np.arange(n_layers)})
    for a, b in [("r_c", "r_a"), ("h_c", "h_a"), ("r_c", "h_c"), ("r_a", "h_a")]:
        df[f"cos_{a}_{b}"] = cos(dirs[a], dirs[b]).numpy()
    for k, (a, b, pos) in SPECS.items():
        df[f"norm_{k}"] = raw[k].norm(dim=-1).numpy()
        sh = split_half(train[a], train[b], pos, cfg["n_split_half"], rng, paired=k in ("r_a", "h_a"))
        df[f"splithalf_{k}_mean"] = sh.mean(0).numpy()
        df[f"splithalf_{k}_std"] = sh.std(0).numpy()
        # held-out separability: project test activations on the direction, AUROC on each task
        for task, (tp, tn) in TASKS.items():
            aucs = []
            for l in range(n_layers):
                x = torch.cat([test[tp][:, l, pos], test[tn][:, l, pos]]) @ dirs[k][l]
                y = np.r_[np.ones(len(test[tp])), np.zeros(len(test[tn]))]
                aucs.append(roc_auc_score(y, x.numpy()) if len(set(y)) == 2 else np.nan)
            df[f"auc_{k}_on_{task}"] = aucs
    df.to_csv(p.run / "directions_layers.csv", index=False)
    torch_save_atomic({**dirs, "raw": raw, "model": cfg["model"]}, p.run / "directions.pt")
    print(f"[directions] {n_layers} layers -> {p.run / 'directions.pt'}")
    mid = n_layers // 2
    print(df.iloc[[n_layers // 4, mid, 3 * n_layers // 4]][
        ["layer", "cos_r_c_r_a", "cos_h_c_h_a", "cos_r_c_h_c", "cos_r_a_h_a",
         "splithalf_r_c_mean", "splithalf_r_a_mean"]].round(3).to_string(index=False))
    return dirs


def main():
    args = parse_args(__doc__)
    compute(load_config(args))


if __name__ == "__main__":
    main()
