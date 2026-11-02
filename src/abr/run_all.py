"""Full pipeline: make_data -> collect -> directions -> ablate_eval -> analyze.

Every stage is resumable (outputs already on disk are skipped), so just re-run after an interruption.
  python -m abr.run_all --preset smoke
  python -m abr.run_all --stages collect directions        # e.g. activations only (cloud)
"""
from __future__ import annotations

from . import ablate_eval, analyze, collect, directions, make_data
from .common import SETS, load_config, parse_args, paths

STAGES = ["make_data", "collect", "directions", "ablate_eval", "analyze"]


def main():
    def extra(p):
        p.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
        p.add_argument("--layer", type=int, default=None)
        p.add_argument("--remake-data", action="store_true")

    args = parse_args(__doc__, extra)
    cfg = load_config(args)
    p = paths(cfg)
    print(f"[run_all] model={cfg['model']} tag={cfg['tag']} run_dir={p.run}")
    for stage in STAGES:
        if stage not in args.stages:
            continue
        print(f"\n===== {stage} =====")
        if stage == "make_data":
            if all(p.set_file(s).exists() for s in SETS) and not args.remake_data:
                print(f"[run_all] data present in {p.data}, skipping (--remake-data to rebuild)")
                continue
            make_data.run(cfg)
        elif stage == "collect":
            collect.run(cfg)
        elif stage == "directions":
            directions.compute(cfg)
        elif stage == "ablate_eval":
            ablate_eval.run(cfg, args.layer)
        elif stage == "analyze":
            analyze.run(cfg)


if __name__ == "__main__":
    main()
