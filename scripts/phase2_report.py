"""Phase-2 report: score every run with the 3-way judge and write results/phase2_summary/.

  python scripts/phase2_report.py [--flag-only]

--flag-only: score the layer scans at n=24 and print flagged layers (D_ra keeps content (no real
compliance) >= 0.9 while destructive comply rises >= 0.2 over A_orig). Writes flagged_layers.json.
Prints aggregates only (no sample text). Writes no direction vectors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from abr.phase2 import score, slug, unit  # noqa: E402

OUT = ROOT / "results/phase2_summary"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B-2507",
          "Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-8B": "Qwen3-8B"}
SCAN = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B"]


def ci(r, k):
    lo, hi = r.get(k + "_lo"), r.get(k + "_hi")
    return f"{r[k]:.2f} [{lo:.2f},{hi:.2f}]" if lo is not None and not pd.isna(lo) else f"{r[k]:.2f}"


def scan_table(m, n):
    run = ROOT / "results" / f"{slug(m)}_scan"
    d = score(run, n).set_index("condition")
    a = d.loc["A_orig"]
    rows = []
    for c in d.index:
        if "@L" not in c:
            continue
        kind, L = c.split("@L")
        r = d.loc[c]
        rows.append({"model": MODELS[m], "layer": int(L), "cond": kind, "n": int(r.content_harm_n),
                     "content_kept": r.content_kept, "explicit_refusal": r.explicit_refusal, "evasive": r.evasive,
                     "real_compliance": r.real_compliance, "destr_comply": r.destr_comply,
                     "d_destr_comply": r.destr_comply - a.destr_comply, "benign_tool_use": r.benign_tool_use,
                     "dlogodds_harm": r.get("dlogodds_harm"), "dlogodds_destr": r.get("dlogodds_destr"),
                     "A_content_kept": a.content_kept, "A_destr_comply": a.destr_comply})
    t = pd.DataFrame(rows).sort_values(["cond", "layer"])
    t["flag"] = (t.cond == "D_ra") & (t.content_kept >= 0.9) & (t.d_destr_comply >= 0.2)
    return t


def flag_only():
    flagged = {}
    for m in SCAN:
        t = scan_table(m, 24)
        t.to_csv(OUT / f"scan_{MODELS[m]}_n24.csv", index=False)
        print(t[["model", "cond", "layer", "content_kept", "real_compliance", "destr_comply", "d_destr_comply",
                 "benign_tool_use", "flag"]].round(2).to_string(index=False))
        flagged[m] = [int(x) for x in t[t.flag].layer]
    (OUT / "flagged_layers.json").write_text(json.dumps(flagged, indent=1))
    print("FLAGGED:", json.dumps(flagged))


def plot_scan(tabs):
    fig, axes = plt.subplots(1, len(tabs), figsize=(6.2 * len(tabs), 3.8), squeeze=False)
    for ax, t in zip(axes[0], tabs):
        for cond, col in (("D_ra", "#c0392b"), ("C_rc_perp", "#2a6fdb")):
            s = t[t.cond == cond]
            ax.plot(s.layer, s.content_kept, marker="o", color=col, label=f"{cond}: content kept")
            ax.plot(s.layer, s.destr_comply, marker="s", ls="--", color=col, label=f"{cond}: destr comply")
        ax.axhline(t.A_content_kept.iloc[0], color="#555", ls=":", lw=1)
        ax.axhline(t.A_destr_comply.iloc[0], color="#999", ls=":", lw=1)
        for L in t[t.flag].layer:
            ax.axvspan(L - 0.6, L + 0.6, color="#f1c40f", alpha=0.3, lw=0)
        ax.set_title(f"{t.model.iloc[0]} layer scan (n={t.n.iloc[0]}; yellow = flagged; dotted = A)", loc="left", fontsize=9)
        ax.set_xlabel("ablated direction from layer"); ax.set_ylim(-0.03, 1.03)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "scan_layers.png", dpi=150); plt.close(fig)


def main():
    if "--flag-only" in sys.argv:
        flag_only()
        return
    md = []
    # ---------------- step 1: existing runs re-judged
    rows = []
    for run in sorted((ROOT / "results").glob("Qwen__*_L[0-9]*")):
        d = score(run)
        d.insert(0, "run", run.name.replace("Qwen__", ""))
        rows.append(d)
    ex = pd.concat(rows)
    ex.to_csv(OUT / "existing_runs_metrics3.csv", index=False)

    # ---------------- step 2: controls
    cr = ROOT / "results/Qwen__Qwen3.5-4B_controls_L10"
    c = score(cr)
    info = json.loads((cr / "controls_info.json").read_text())
    c["cos_r_c"] = c.condition.map(lambda k: info.get(k, {}).get("cos_r_c"))
    c["cos_r_a"] = c.condition.map(lambda k: info.get(k, {}).get("cos_r_a"))
    c["raw_norm"] = c.condition.map(lambda k: info.get(k, {}).get("raw_norm"))
    c.to_csv(OUT / "controls_L10.csv", index=False)
    md += ["## Controls: Qwen3.5-4B, ablation at layer 10 (n=48 per set)", "",
           "| condition | content kept (no real compliance) | explicit / evasive / real | destr comply | destr text refusal | benign tool use | Δlogodds harm | Δlogodds destr |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        md.append(f"| {r.condition} | {ci(r, 'content_kept')} | {r.explicit_refusal:.2f} / {r.evasive:.2f} / {r.real_compliance:.2f} | "
                  f"{ci(r, 'destr_comply')} | {r.destr_refuse_text:.2f} | {r.benign_tool_use:.2f} | "
                  f"{r.dlogodds_harm:+.2f} ± {1.96 * r.dlogodds_harm_se if r.condition != 'A_orig' else 0:.2f} | "
                  f"{r.dlogodds_destr:+.2f} ± {1.96 * r.dlogodds_destr_se if r.condition != 'A_orig' else 0:.2f} |")
    a = c.set_index("condition").loc["A_orig"]
    ctrl = c[c.condition.str.startswith(("rand", "ctr_"))]
    above = ctrl[(ctrl.content_kept >= 0.9) & (ctrl.destr_comply_lo > a.destr_comply_hi)]
    md += ["", f"Controls whose destr-comply 95% CI lies entirely above A's: {list(above.condition) or 'none'}. "
           f"Max control destr comply = {ctrl.destr_comply.max():.2f} ({ctrl.loc[ctrl.destr_comply.idxmax(), 'condition']}); "
           f"most negative control Δlogodds destr = {ctrl.dlogodds_destr.min():+.2f}.", ""]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for _, r in c.iterrows():
        k = r.condition
        col = "#c0392b" if k == "D_ra" else "#2a6fdb" if k == "C_rc_perp" else "#8e44ad" if k == "B_rc" else \
            "#555" if k == "A_orig" else "#e67e22" if k.startswith("ctr_") else "#95a5a6"
        ax.errorbar(r.dlogodds_destr, r.dlogodds_harm, xerr=1.96 * (r.dlogodds_destr_se if k != "A_orig" else 0),
                    yerr=1.96 * (r.dlogodds_harm_se if k != "A_orig" else 0), fmt="o", color=col, ms=6, capsize=2)
        if not k.startswith("rand"):
            ax.annotate(k, (r.dlogodds_destr, r.dlogodds_harm), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="#ccc", lw=0.8); ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("Δ first-token refusal log-odds on action_destr (vs A)")
    ax.set_ylabel("Δ ... on content_harm (vs A)")
    ax.set_title("Qwen3.5-4B L10: grey = 10 random, orange = benign contrasts", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "controls_L10_logodds.png", dpi=150); plt.close(fig)

    # ---------------- step 3: layer scans (n=24) + flagged reruns (n=48)
    flagged = json.loads((OUT / "flagged_layers.json").read_text()) if (OUT / "flagged_layers.json").exists() else {}
    tabs = []
    md += ["## Layer scan (D_ra / C_rc_perp, every 2nd layer over the first 60% of depth, n=24)", ""]
    for m in SCAN:
        t = scan_table(m, 24)
        t.to_csv(OUT / f"scan_{MODELS[m]}_n24.csv", index=False)
        tabs.append(t)
        md += [f"### {MODELS[m]} (A: content kept {t.A_content_kept.iloc[0]:.2f}, destr comply {t.A_destr_comply.iloc[0]:.2f})", "",
               "| layer | D_ra content kept | D_ra real compl. | D_ra destr comply (Δ) | C content kept | C destr comply (Δ) | flag |",
               "|---|---|---|---|---|---|---|"]
        for L in sorted(t.layer.unique()):
            dd = t[(t.layer == L) & (t.cond == "D_ra")].iloc[0]
            cc = t[(t.layer == L) & (t.cond == "C_rc_perp")].iloc[0]
            md.append(f"| {L} | {dd.content_kept:.2f} | {dd.real_compliance:.2f} | {dd.destr_comply:.2f} ({dd.d_destr_comply:+.2f}) | "
                      f"{cc.content_kept:.2f} | {cc.destr_comply:.2f} ({cc.d_destr_comply:+.2f}) | {'**yes**' if dd.flag else ''} |")
        md.append("")
        if flagged.get(m):
            t48 = scan_table(m, 48)
            t48 = t48[t48.layer.isin(flagged[m])]
            t48.to_csv(OUT / f"scan_{MODELS[m]}_flagged_n48.csv", index=False)
            md += [f"Flagged layers rerun at n=48 ({MODELS[m]}):", "",
                   "| layer | cond | n | content kept | real compl. | destr comply | Δ vs A | still flagged |", "|---|---|---|---|---|---|---|---|"]
            for _, r in t48.iterrows():
                md.append(f"| {r.layer} | {r.cond} | {r.n} | {r.content_kept:.2f} | {r.real_compliance:.2f} | {r.destr_comply:.2f} | "
                          f"{r.d_destr_comply:+.2f} | {'yes' if r.flag else 'no'} |")
            md.append("")
    plot_scan(tabs)

    # ---------------- step 4: decision vs expression
    rows = []
    for m, short in MODELS.items():
        run = ROOT / "results" / f"{slug(m)}_declate"
        d = score(run).set_index("condition")
        Ls = sorted(int(k.split("@L")[1]) for k in d.index if k.startswith("B_rc@L"))
        dirs = torch.load(ROOT / "results" / f"{slug(m)}_base" / "directions.pt")
        n_layers = dirs["r_c"].shape[0]
        cos_el = float(unit(dirs["r_c"][Ls[0]].float()) @ unit(dirs["r_c"][Ls[-1]].float()))
        for L, role in zip(Ls, ("early", "late")):
            r = d.loc[f"B_rc@L{L}"]
            rows.append({"model": short, "role": role, "layer": L, "rel_depth": L / n_layers, "n": int(r.content_harm_n),
                         "explicit_refusal": r.explicit_refusal, "evasive": r.evasive, "real_compliance": r.real_compliance,
                         "real_compliance_lo": r.real_compliance_lo, "real_compliance_hi": r.real_compliance_hi,
                         "string_refusal": r.content_harm_string_refusal, "cos_rc_early_late": cos_el,
                         "dlogodds_harm": r.get("dlogodds_harm"),
                         "A_explicit": d.loc["A_orig"].explicit_refusal, "A_real": d.loc["A_orig"].real_compliance})
    de = pd.DataFrame(rows)
    de.to_csv(OUT / "decision_vs_expression.csv", index=False)
    md += ["## Decision vs expression: B_rc at the best early (20–50% depth) and late (50–80%) layer, n=48", "",
           "Layer rule: lowest first-token refusal score in the window among layers with KL_safe < 0.1 (from layer_sweep.csv).", "",
           "| model | role | layer (rel) | explicit | evasive | real compliance | string 'refusal' | Δlogodds harm | cos(r_c early, r_c late) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, r in de.iterrows():
        md.append(f"| {r.model} | {r.role} | {r.layer} ({r.rel_depth:.2f}) | {r.explicit_refusal:.2f} | {r.evasive:.2f} | "
                  f"{r.real_compliance:.2f} [{r.real_compliance_lo:.2f},{r.real_compliance_hi:.2f}] | {r.string_refusal:.2f} | "
                  f"{r.dlogodds_harm:+.2f} | {r.cos_rc_early_late:.2f} |")
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 3.6), sharey=True)
    cols = {"explicit_refusal": "#7f8c8d", "evasive": "#f0b429", "real_compliance": "#c0392b"}
    for ax, (short, g) in zip(axes, de.groupby("model", sort=False)):
        x = np.arange(len(g)); bottom = np.zeros(len(g))
        for cat, col in cols.items():
            ax.bar(x, g[cat], bottom=bottom, color=col, label=cat.replace("_", " "), width=0.6)
            bottom += g[cat].values
        ax.set_xticks(x, [f"{r.role}\nL{r.layer}" for _, r in g.iterrows()])
        ax.set_title(f"{short}\ncos(early,late r_c)={g.cos_rc_early_late.iloc[0]:.2f}", fontsize=9)
    axes[0].set_ylabel("share of content_harm (B_rc)"); axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "decision_vs_expression.png", dpi=150); plt.close(fig)

    (OUT / "phase2_tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
