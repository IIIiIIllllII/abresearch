"""Debug: render one sample per set and show which tokens t_inst / t_post_inst point at."""
import sys

from transformers import AutoTokenizer

from abr.common import SETS, load_config, load_split, parse_args, paths
from abr.prompts import encode_batch

cfg = load_config(parse_args(__doc__))
tok = AutoTokenizer.from_pretrained(cfg["model"])
tok.padding_side = "left"
p = paths(cfg)
for s in SETS:
    smp = load_split(p, s, "train")
    if s.startswith("action"):  # prefer a sample with tool-call history
        smp = sorted(smp, key=lambda r: r["history_id"] not in (1, 2, 3, 7))
    smp = smp[:2]
    enc, texts, ti, tp = encode_batch(tok, smp, cfg["chat_template_kwargs"], "cpu")
    ids = enc["input_ids"][0]
    print(f"===== {s} =====")
    print(texts[0][texts[0].find("</tools>"):] if s.startswith("action") else texts[0])
    print(f"t_inst -> {tok.decode(ids[ti[0]-3:ti[0]+1])!r} | t_post_inst -> {tok.decode(ids[tp[0]-3:tp[0]+1])!r}")
