"""Collect residual-stream activations at t_inst and t_post_inst for every layer, every set/split.

Output per set/split: activations/<set>_<split>.pt = {"ids", "acts": float16 [N, L, 2, d], "positions"}
(positions axis order = common.POSITIONS). Existing files are skipped, so the step is resumable.
"""
from __future__ import annotations

import torch

from .common import (POSITIONS, SETS, SPLITS, batched, input_device, load_config, load_model,
                     load_split, parse_args, paths, torch_save_atomic)
from .hooks import ablate_direction, capture_residual
from .prompts import encode_batch


@torch.no_grad()
def collect_acts(model, tok, samples, cfg, direction=None) -> torch.Tensor:
    """[N, L, 2, d] float16 activations (optionally under directional ablation)."""
    dev = input_device(model)
    out = []
    for batch in batched(samples, cfg["batch_size"]):
        enc, _, t_inst, t_post = encode_batch(tok, batch, cfg["chat_template_kwargs"], dev)
        store: dict = {}
        with ablate_direction(model, direction), capture_residual(model, store):
            model(**enc, use_cache=False)
        rows = torch.arange(len(batch))
        per_layer = []
        for l in range(len(store)):
            h = store[l]
            idx_i = torch.tensor(t_inst, device=h.device)
            idx_p = torch.tensor(t_post, device=h.device)
            r = rows.to(h.device)
            per_layer.append(torch.stack([h[r, idx_i], h[r, idx_p]], dim=1).float().cpu())
        out.append(torch.stack(per_layer, dim=1).half())
    return torch.cat(out)


def run(cfg: dict):
    p = paths(cfg)
    todo = [(s, sp) for s in SETS for sp in SPLITS if not (p.acts / f"{s}_{sp}.pt").exists()]
    if not todo:
        print("[collect] all activations present, skipping")
        return
    model, tok = load_model(cfg)
    for s, sp in todo:
        samples = load_split(p, s, sp)
        acts = collect_acts(model, tok, samples, cfg)
        torch_save_atomic({"ids": [x["id"] for x in samples], "acts": acts, "positions": POSITIONS,
                           "model": cfg["model"]}, p.acts / f"{s}_{sp}.pt")
        print(f"[collect] {s}/{sp}: {tuple(acts.shape)}")


def main():
    run(load_config(parse_args(__doc__)))


if __name__ == "__main__":
    main()
