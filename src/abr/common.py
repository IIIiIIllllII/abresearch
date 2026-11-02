"""Shared utilities: config, paths, device, model loading, layer access, IO."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

SETS = ["content_harm", "content_safe", "action_destr", "action_benign"]
SPLITS = ["train", "test"]
POSITIONS = ["t_inst", "t_post_inst"]


# ---------------------------------------------------------------- config / paths

def parse_args(desc: str, extra=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--preset", default=None, help="name of a preset in config.yaml (e.g. smoke)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override config values")
    if extra:
        extra(p)
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(Path(args.config).read_text())
    presets = cfg.pop("presets", {}) or {}
    if args.preset:
        if args.preset not in presets:
            raise SystemExit(f"unknown preset {args.preset!r}; available: {list(presets)}")
        cfg.update(presets[args.preset])
        cfg["tag"] = args.preset
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg.setdefault("tag", None)
    return cfg


@dataclass
class Paths:
    data: Path
    run: Path

    @property
    def acts(self) -> Path:
        return self.run / "activations"

    @property
    def gens(self) -> Path:
        return self.run / "generations"

    @property
    def proj(self) -> Path:
        return self.run / "h_proj"

    def set_file(self, name: str) -> Path:
        return self.data / f"{name}.jsonl"


def model_slug(name: str) -> str:
    return name.replace("/", "__")


def paths(cfg: dict) -> Paths:
    tag = cfg.get("tag")
    run = model_slug(cfg["model"]) + (f"_{tag}" if tag else "")
    p = Paths(data=ROOT / cfg["data_dir"] / (tag or "full"), run=ROOT / cfg["results_dir"] / run)
    for d in (p.data, p.run, p.acts, p.gens, p.proj):
        d.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- IO

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            print(f"[io] dropping truncated last line of {path} (interrupted write)")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        write_jsonl(path, read_jsonl(path))  # crash left a partial line: rewrite without it
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def torch_save_atomic(obj, path: Path) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_split(p: Paths, name: str, split: str) -> list[dict]:
    rows = [r for r in read_jsonl(p.set_file(name)) if r["split"] == split]
    if not rows:
        raise SystemExit(f"no data for {name}/{split} in {p.data}; run make_data first")
    return rows


# ---------------------------------------------------------------- device / model

def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = cfg.get("device") or detect_device()
    dtype = getattr(torch, DTYPES[cfg["dtype"]])
    if device == "cpu" and dtype != torch.float32:
        print(f"[model] cpu device: keeping {cfg['dtype']} (set dtype=fp32 if ops are unsupported)")
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(dtype=dtype)
    if cfg.get("device_map"):
        kw["device_map"] = cfg["device_map"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kw)
    if not cfg.get("device_map"):
        model.to(device)
    model.eval()
    model.requires_grad_(False)
    print(f"[model] {cfg['model']} dtype={dtype} device={device} device_map={cfg.get('device_map')} "
          f"layers={len(get_layers(model))} d_model={model.config.get_text_config().hidden_size}")
    return model, tok


def input_device(model):
    return model.get_input_embeddings().weight.device


def get_layers(model):
    """Return the ModuleList of decoder blocks for common HF architectures."""
    candidates = [
        "model.layers",                     # llama / qwen / mistral / gemma
        "model.language_model.layers",      # gemma3 / multimodal wrappers
        "model.model.layers",
        "transformer.h",                    # gpt2 / falcon style
        "gpt_neox.layers",
    ]
    for path in candidates:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise RuntimeError("could not locate decoder layers; add the attribute path to get_layers()")


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
