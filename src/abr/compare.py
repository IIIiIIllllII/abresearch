"""Cross-model comparison (CPU only). Reads results/<run>/{directions_layers.csv, metrics.csv, conditions.json}.

  python -m abr.compare                       # every non-smoke run under results/
  python -m abr.compare --runs Qwen__Qwen3.5-4B Qwen__Qwen3-4B-Instruct-2507

Writes results/comparison/{summary.csv, comparison_geometry.png, comparison_ablation.png}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .analyze import C, INK2, SURF, plt  # shared palette + rcParams
from .common import ROOT

# Role of each model in the study; unknown runs are shown as "extra".
ROLES = {
    "Qwen__Qwen3.5-4B": "main (4B, new gen)",
    "Qwen__Qwen3-4B-Instruct-2507": "control (4B, prev gen)",
    "Qwen__Qwen3-1.7B": "small baseline (1.7B, prev gen)",
}
ORDER = list(ROLES)
CONDS = ["A_orig", "B_rc", "C_rc_perp"]


def load_run(run_dir) -> dict | None:
    need = ["directions_layers.csv", "metrics.csv", "conditions.json"]
    if not all((run_dir / f).exists() for f in need):
        print(f"[compare] skipping {run_dir.name}: missing {[f for f in need if not (run_dir / f).exists()]}")
        return None
    return {"name": run_dir.name, "dirs": pd.read_csv(run_dir / "directions_layers.csv"),
            "metrics": pd.read_csv(run_dir / "metrics.csv").set_index("condition"),
            "info": json.loads((run_dir / "conditions.json").read_text())}


def summarize(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        d, m, L = r["dirs"], r["metrics"], r["info"]["layer"]
        at = d.set_index("layer").loc[L]
        ceil = np.sqrt(max(at.splithalf_r_c_mean, 0) * max(at.splithalf_r_a_mean, 0))
        row = {"run": r["name"], "role": ROLES.get(r["name"], "extra"), "n_layers": len(d), "layer": L,
               "cos_rc_ra@L": at.cos_r_c_r_a, "ceiling_rc_ra@L": ceil, "cos_rc_ra_max": d.cos_r_c_r_a.max(),
               "cos_hc_ha@L": at.cos_h_c_h_a, "cos_rc_hc@L": at.cos_r_c_h_c, "cos_ra_ha@L": at.cos_r_a_h_a,
               "auc_rc_on_action@L": at.auc_r_c_on_action, "auc_ra_on_content@L": at.auc_r_a_on_content}
        for c in CONDS:
            if c not in m.index:
                continue
            row[f"{c}:harm_refusal"] = m.loc[c, "content_harm_refusal_rate"]
            row[f"{c}:safe_refusal"] = m.loc[c, "content_safe_refusal_rate"]
            row[f"{c}:destr_comply"] = m.loc[c, "action_destr_comply"]
            row[f"{c}:destr_clear_comply"] = m.loc[c, "action_destr_clear_comply"]
            row[f"{c}:benign_tool_use"] = m.loc[c, "action_benign_tool_use_rate"]
            row[f"{c}:h_a_auc"] = m.loc[c, "h_a_proj_auc"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_geometry(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for i, r in enumerate(runs):
        d, color = r["dirs"], C[i]
        depth = d.layer / (len(d) - 1)
        lab = f"{r['name'].split('__')[-1]}  [{ROLES.get(r['name'], 'extra')}]"
        ceil = np.sqrt(d.splithalf_r_c_mean.clip(0) * d.splithalf_r_a_mean.clip(0))
        axes[0].plot(depth, d.cos_r_c_r_a, color=color, label=lab)
        axes[0].plot(depth, ceil, color=color, linestyle=":", linewidth=1.2)
        axes[1].plot(depth, d.auc_r_c_on_action, color=color, label=lab)
        axes[2].plot(depth, d.cos_r_a_h_a, color=color, label=lab)
        Lsel = r["info"]["layer"] / (len(d) - 1)
        for ax in axes:
            ax.axvline(Lsel, color=color, linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].set_title("cos(r_c, r_a)   (dotted: noise ceiling)", loc="left")
    axes[1].set_title("r_c projection: AUROC on destr vs benign actions", loc="left")
    axes[1].axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    axes[2].set_title("cos(r_a, h_a)  action refusal vs harmfulness", loc="left")
    for ax in axes:
        ax.set_xlabel("relative depth (layer / last layer); dashed = selected layer")
    axes[0].axhline(0, color=INK2, linewidth=0.8)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "comparison_geometry.png", dpi=150)
    plt.close(fig)


def plot_ablation(runs, out):
    panels = [("content_harm_refusal_rate", "Content refusal (harmful prompts)"),
              ("action_destr_comply", "Destructive actions executed (comply)"),
              ("action_benign_tool_use_rate", "Benign tool use (over-suppression check)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    x = np.arange(len(CONDS))
    w = 0.8 / len(runs)
    for ax, (col, title) in zip(axes, panels):
        for i, r in enumerate(runs):
            m = r["metrics"]
            v = [m.loc[c, col] if c in m.index else np.nan for c in CONDS]
            off = (i - (len(runs) - 1) / 2) * w
            ax.bar(x + off, v, w * 0.92, color=C[i], edgecolor=SURF, linewidth=1.5,
                   label=f"{r['name'].split('__')[-1]}  [{ROLES.get(r['name'], 'extra')}]")
        ax.set_xticks(x, CONDS)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left")
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "comparison_ablation.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="*", default=None)
    args = p.parse_args()
    res = ROOT / "results"
    names = args.runs or sorted(d.name for d in res.iterdir()
                                if d.is_dir() and not d.name.endswith("_smoke") and d.name != "comparison")
    names = sorted(names, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER))
    runs = [r for r in (load_run(res / n) for n in names) if r]
    if not runs:
        raise SystemExit("no complete runs to compare")
    out = res / "comparison"
    out.mkdir(exist_ok=True)
    s = summarize(runs)
    s.to_csv(out / "summary.csv", index=False)
    plot_geometry(runs, out)
    plot_ablation(runs, out)
    print(s.set_index("run").T.round(3).to_string())
    print(f"[compare] -> {out}")


if __name__ == "__main__":
    main()
