"""Run abr's layer sweep in the base run dir and print the auto-selected layer."""
import sys, torch
from abr.common import load_config, parse_args, paths, load_model
from abr.ablate_eval import select_layer
cfg = load_config(parse_args("sweep"))
p = paths(cfg)
f = p.run / "auto_layer.txt"
if f.exists():
    print("[sweep] auto layer", f.read_text()); sys.exit(0)
dirs = torch.load(p.run / "directions.pt")
model, tok = load_model(cfg)
l = select_layer(model, tok, cfg, p, dirs)
f.write_text(str(l)); print("[sweep] auto layer", l)
