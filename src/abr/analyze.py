"""Analysis (CPU only): judge generations, compute metrics, draw plots.

Needs only results/<run>/{activations or directions.pt, generations/, h_proj/, conditions.json};
any missing piece is skipped, so you can analyze activations alone (e.g. pulled back from the cloud).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .common import SETS, load_config, parse_args, paths, read_jsonl, write_jsonl  # noqa: E402
from .judge import ACTION_CATEGORIES, make_judge  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode); fixed order, never cycled.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CAT_COLORS = dict(zip(ACTION_CATEGORIES, [C[7], C[1], C[3], C[2], "#a3a29c"]))
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- direction geometry

def plot_directions(p, df: pd.DataFrame):
    x = df.layer
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    # cross-cosine series: (column, label, color, (a, b)) -- dotted line of the same color is the noise
    # ceiling sqrt(rel_a * rel_b): roughly what cos would be if a and b were the SAME direction.
    panels = [
        ("Same role, different domain", [("cos_r_c_r_a", "cos(r_c, r_a)  refusal", C[0], ("r_c", "r_a")),
                                         ("cos_h_c_h_a", "cos(h_c, h_a)  harmfulness", C[1], ("h_c", "h_a"))]),
        ("Refusal vs harmfulness", [("cos_r_c_h_c", "cos(r_c, h_c)  content", C[2], ("r_c", "h_c")),
                                    ("cos_r_a_h_a", "cos(r_a, h_a)  action", C[3], ("r_a", "h_a"))]),
        ("Baseline: split-half reliability", [("splithalf_r_c_mean", "r_c", C[0], None),
                                              ("splithalf_r_a_mean", "r_a", C[4], None),
                                              ("splithalf_h_c_mean", "h_c", C[2], None),
                                              ("splithalf_h_a_mean", "h_a", C[3], None)]),
    ]
    for ax, (title, series) in zip(axes, panels):
        for col, label, color, pair in series:
            ax.plot(x, df[col], color=color, label=label)
            sd = col.replace("_mean", "_std")
            if sd in df and sd != col:
                ax.fill_between(x, df[col] - df[sd], df[col] + df[sd], color=color, alpha=0.15, linewidth=0)
            if pair:
                a, b = (df[f"splithalf_{k}_mean"].clip(0) for k in pair)
                ax.plot(x, np.sqrt(a * b), color=color, linestyle=":", linewidth=1.4,
                        label=f"  ceiling √(rel {pair[0]}·rel {pair[1]})")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("layer")
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("cosine similarity")
    lo = min(df.filter(regex="^(cos_|splithalf_.*_mean)").min().min() - 0.05, -0.2)
    axes[0].set_ylim(max(lo, -1.0), 1.15)
    fig.tight_layout()
    fig.savefig(p.run / "cosine_by_layer.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, task in zip(axes, ["content", "action"]):
        for k, color in zip(["r_c", "r_a", "h_c", "h_a"], [C[0], C[4], C[2], C[3]]):
            ax.plot(x, df[f"auc_{k}_on_{task}"], color=color, label=k)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_title(f"Held-out AUROC on the {task} task", loc="left")
        ax.set_xlabel("layer")
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("AUROC (projection on direction)")
    fig.tight_layout()
    fig.savefig(p.run / "auroc_by_layer.png", dpi=150)
    plt.close(fig)


def plot_sweep(p):
    f = p.run / "layer_sweep.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(df.layer, df.refusal_score_harm, color=C[0], marker="o", markersize=4, label="r_c ablated")
    axes[0].axhline(df.refusal_score_harm_base.iloc[0], color=INK2, linestyle=":", linewidth=1.2, label="no ablation")
    axes[0].set_title("Refusal score on content_harm (first token, log-odds)", loc="left")
    axes[0].legend(fontsize=8)
    axes[1].plot(df.layer, df.kl_safe, color=C[1], marker="o", markersize=4)
    axes[1].set_title("KL on content_safe (first token)", loc="left")
    for ax in axes:
        ax.set_xlabel("direction taken from layer")
    fig.tight_layout()
    fig.savefig(p.run / "layer_sweep.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- generations

def judge_all(p, cfg, conds: list[str]) -> pd.DataFrame:
    judge = make_judge(cfg)
    samples = {}
    for s in SETS:
        for r in read_jsonl(p.set_file(s)):
            samples[r["id"]] = r
    rows = []
    for cond in conds:
        out = []
        for s in SETS:
            for g in read_jsonl(p.gens / cond / f"{s}.jsonl"):
                smp = samples[g["id"]]
                j = judge.judge(smp, g["text"])
                rec = {"condition": cond, "id": g["id"], "set": s, "ambiguous": smp.get("ambiguous", False),
                       "source": smp.get("source"), "request": smp["messages"][-1]["content"],
                       "judge": judge.name, **j, "text": g["text"]}
                out.append(rec)
        if out:
            write_jsonl(p.run / f"samples_{cond}.jsonl", out)
        rows += out
    return pd.DataFrame(rows)


def proj_stats(p, cond: str, layer: int) -> dict:
    f = p.proj / f"{cond}.npz"
    if not f.exists():
        return {}
    z = np.load(f)
    out = {}
    for kind in ("proj", "proj_perp"):
        if f"action_destr__{kind}" not in z:
            continue
        a, b = z[f"action_destr__{kind}"][:, layer], z[f"action_benign__{kind}"][:, layer]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if len(a) > 1 else np.nan
        out[f"h_a_{kind}_destr_mean"] = a.mean()
        out[f"h_a_{kind}_benign_mean"] = b.mean()
        out[f"h_a_{kind}_dprime"] = (a.mean() - b.mean()) / pooled if pooled else np.nan
        out[f"h_a_{kind}_auc"] = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    return out


def metrics(df: pd.DataFrame, p, conds, layer) -> pd.DataFrame:
    rows = []
    for cond in conds:
        d = df[df.condition == cond]
        m = {"condition": cond}
        for s in ("content_harm", "content_safe"):
            x = d[d.set == s]
            k, n = int(x.refused.astype(bool).sum()), len(x)
            m[f"{s}_refusal_rate"] = k / n if n else np.nan
            m[f"{s}_refusal_ci_lo"], m[f"{s}_refusal_ci_hi"] = wilson(k, n)
            m[f"{s}_n"] = n
        destr = d[d.set == "action_destr"]
        tmpl = destr.source == "tmpl"
        groups = {"action_destr": destr, "action_destr_clear": destr[tmpl & ~destr.ambiguous.astype(bool)],
                  "action_destr_ambiguous": destr[tmpl & destr.ambiguous.astype(bool)],
                  "action_destr_agentharm": destr[destr.source == "agentharm"]}
        for s, x in groups.items():
            m[f"{s}_n"] = len(x)
            for c in ACTION_CATEGORIES:
                m[f"{s}_{c}"] = (x.category == c).mean() if len(x) else np.nan
        x = d[d.set == "action_benign"]
        n = len(x)
        k = int(x.tool_call.astype(bool).sum()) if n else 0
        m["action_benign_n"] = n
        m["action_benign_tool_use_rate"] = k / n if n else np.nan
        m["action_benign_tool_ci_lo"], m["action_benign_tool_ci_hi"] = wilson(k, n)
        for c in ("refuse_text", "clarify", "other"):
            m[f"action_benign_{c}"] = (x.category == c).mean() if n else np.nan
        m.update(proj_stats(p, cond, layer))
        rows.append(m)
    return pd.DataFrame(rows)


def plot_eval(p, mdf: pd.DataFrame, conds, layer):
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1.8, 1]})

    ax = axes[0]
    w = 0.38
    for i, (s, color) in enumerate([("content_harm", C[0]), ("content_safe", C[1])]):
        r = mdf[f"{s}_refusal_rate"].values
        err = np.vstack([r - mdf[f"{s}_refusal_ci_lo"], mdf[f"{s}_refusal_ci_hi"] - r])
        ax.bar(x + (i - 0.5) * w, r, w * 0.94, color=color, label=s, yerr=err, ecolor=INK2, capsize=3)
    ax.set_xticks(x, conds)
    ax.set_ylim(0, 1.05)
    ax.set_title("Content refusal rate (95% Wilson CI)", loc="left")
    ax.legend(fontsize=8)

    ax = axes[1]
    groups = [("action_destr_clear", "clear"), ("action_destr_ambiguous", "ambig"), ("action_destr_agentharm", "AH")]
    groups = [g for g in groups if mdf[f"{g[0]}_n"].max() > 0]
    w = 0.8 / len(groups)
    for gi, (g, short) in enumerate(groups):
        off = (gi - (len(groups) - 1) / 2) * w
        bottom = np.zeros(len(conds))
        for c in ACTION_CATEGORIES:
            v = np.nan_to_num(mdf[f"{g}_{c}"].values)
            ax.bar(x + off, v, w * 0.94, bottom=bottom, color=CAT_COLORS[c],
                   label=c if gi == 0 else None, edgecolor=SURF, linewidth=1.5)
            bottom += v
        for xi in x:
            ax.text(xi + off, 1.02, f"{short}\nn={int(mdf[f'{g}_n'].iloc[0])}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, conds)
    ax.set_ylim(0, 1.15)
    ax.set_title("action_destr outcome by subset", loc="left")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))

    ax = axes[2]
    r = mdf.action_benign_tool_use_rate.values
    err = np.vstack([r - mdf.action_benign_tool_ci_lo, mdf.action_benign_tool_ci_hi - r])
    ax.bar(x, r, 0.6, color=C[2], yerr=err, ecolor=INK2, capsize=3)
    ax.set_xticks(x, conds)
    ax.set_ylim(0, 1.05)
    ax.set_title("action_benign tool-use rate (over-suppression)", loc="left")
    fig.tight_layout()
    fig.savefig(p.run / "eval_conditions.png", dpi=150)
    plt.close(fig)

    # h_a projection distributions
    files = [(c, p.proj / f"{c}.npz") for c in conds if (p.proj / f"{c}.npz").exists()]
    if not files:
        return
    fig, axes = plt.subplots(2, len(files), figsize=(4 * len(files), 6), sharex="row", squeeze=False)
    for j, (cond, f) in enumerate(files):
        z = np.load(f)
        for i, kind in enumerate(("proj", "proj_perp")):
            ax = axes[i, j]
            key = kind if f"action_destr__{kind}" in z else "proj"
            a, b = z[f"action_destr__{key}"][:, layer], z[f"action_benign__{key}"][:, layer]
            bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 20)
            ax.hist(b, bins=bins, color=C[2], histtype="step", linewidth=2, label="action_benign")
            ax.hist(a, bins=bins, color=C[7], histtype="step", linewidth=2, label="action_destr")
            what = "h_a" if key == "proj" else "h_a perp ablated dir"
            if kind == "proj_perp" and key == "proj":
                what += " (no ablation)"
            ax.set_title(f"{cond}: t_inst · {what}", loc="left", fontsize=9)
            ax.set_ylabel("count" if j == 0 else "")
            if j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Harmfulness signal (h_a) at t_inst, layer {layer}, test set", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(p.run / "h_a_projection.png", dpi=150)
    plt.close(fig)


def run(cfg: dict):
    p = paths(cfg)

    if not (p.run / "directions_layers.csv").exists():
        if not any(p.acts.glob("*.pt")):
            raise SystemExit(f"nothing to analyze in {p.run}")
        from .directions import compute
        compute(cfg)
    df = pd.read_csv(p.run / "directions_layers.csv")
    plot_directions(p, df)
    plot_sweep(p)
    print(f"[analyze] direction plots -> {p.run}")

    cj = p.run / "conditions.json"
    if not cj.exists():
        print("[analyze] no conditions.json yet (ablate_eval not run); stopping after direction analysis")
        return
    info = json.loads(cj.read_text())
    conds, layer = info["conditions"], info["layer"]
    jdf = judge_all(p, cfg, conds)
    if jdf.empty:
        print("[analyze] no generations yet")
        return
    mdf = metrics(jdf, p, conds, layer)
    mdf.insert(1, "layer", layer)
    mdf.insert(2, "judge", jdf.judge.iloc[0])
    mdf.to_csv(p.run / "metrics.csv", index=False)
    plot_eval(p, mdf, conds, layer)
    show = ["condition", "content_harm_refusal_rate", "content_safe_refusal_rate",
            "action_destr_comply", "action_destr_refuse_text", "action_destr_clarify", "action_destr_safe_alt",
            "action_benign_tool_use_rate", "h_a_proj_auc", "h_a_proj_perp_auc"]
    print(mdf[[c for c in show if c in mdf]].round(3).to_string(index=False))
    print(f"[analyze] metrics.csv, samples_*.jsonl, eval_conditions.png, h_a_projection.png -> {p.run}")


def main():
    run(load_config(parse_args(__doc__)))


if __name__ == "__main__":
    main()
