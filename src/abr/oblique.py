"""Beyond r_c vs r_c_perp: oblique partial ablation, a learned direction, and probe-gated ablation.

All at the layer selected by ablate_eval (results/<run>/conditions.json), hooks only, weights untouched.
Outputs go to results/<run>/oblique/.

1. sweep        d(theta) = unit(cos(theta) r_a + sin(theta) r_c_perp), x <- x - alpha (x.d) d at every
                block input/output. theta in oblique_thetas (deg), alpha in oblique_alphas. alpha=0 is the
                unablated model ("base", shared by all theta). r_c itself sits at theta_rc = acos(cos(r_c, r_a))
                in this plane, so it is swept too ("rc_*"): rc_a100 = full r_c removal = reference point B.
2. learn        d is a trainable unit vector (init r_c). Loss at t_post_inst (first-token logits), train split:
                  softplus(refusal log-odds on content_harm)                      -> lower content refusal
                + learn_lambda * (refusal log-odds on action_destr - original)^2  -> keep action refusal level
                Gradient flows only into d (model frozen, hook is differentiable). Logs cos with r_c, r_a, r_c_perp.
3. conditional  1-D logistic probe on the t_inst projection onto h_a (layer L), fit on action_destr vs
                action_benign train. Remove r_c only for prompts with probe p < threshold. The gate is computed
                in a clean pass (it cannot depend on activations it would itself ablate), so each prompt's
                output is exactly the base output (gate closed) or the rc_a100 output (gate open): the
                threshold sweep is composed from those two generation sets, no extra generation needed.
4. report       judge (no command is ever executed), paired bootstrap 95% CIs, signed distance from the
                diagonal base->rc_a100, plots + CSVs.

Run order on the GPU: base and rc_a100 first (needed by everything), then learning, then the rest of the
sweep at alpha 1, 0.5, 0.25, 0.75. Every generation file is append-only and resumes per sample.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import torch

from .common import (POSITIONS, append_jsonl, batched, input_device, load_config, load_split, parse_args,
                     paths, read_jsonl, torch_save_atomic, write_jsonl)
from .directions import load_acts, unit

EVAL_SETS = ["content_harm", "action_destr", "action_benign"]
CAUTION = ("refuse_text", "clarify", "safe_alt")
INST = POSITIONS.index("t_inst")

DEFAULTS = {
    "oblique_dir": "oblique",
    "oblique_thetas": [0, 15, 30, 45, 60, 75, 90],
    "oblique_alphas": [0, 0.25, 0.5, 0.75, 1.0],
    "oblique_n": 50,
    "learn_steps": 200,
    "learn_lr": 1e-3,
    "learn_bsz": 4,
    "learn_lambda": 1.0,
    "probe_thresholds": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98],
    "n_boot": 2000,
}


# ---------------------------------------------------------------- setup

class Ctx:
    def __init__(self, cfg):
        for k, v in DEFAULTS.items():
            cfg.setdefault(k, v)
        self.cfg = cfg
        self.p = paths(cfg)
        self.out = self.p.run / cfg["oblique_dir"]
        self.gens = self.out / "generations"
        self.gens.mkdir(parents=True, exist_ok=True)
        cj = self.p.run / "conditions.json"
        if cfg.get("layer") is not None:
            self.layer = int(cfg["layer"])
        elif cj.exists():
            self.layer = json.loads(cj.read_text())["layer"]
        else:
            raise SystemExit("no layer: run ablate_eval first (conditions.json) or pass --set layer=N")
        self.dirs = torch.load(self.p.run / "directions.pt")
        L = self.layer
        self.r_c, self.r_a, self.h_a = (self.dirs[k][L].float() for k in ("r_c", "r_a", "h_a"))
        self.r_perp = unit(self.r_c - (self.r_c @ self.r_a) * self.r_a)
        self.theta_rc = math.degrees(math.atan2(float(self.r_c @ self.r_perp), float(self.r_c @ self.r_a)))
        self.test = {s: load_split(self.p, s, "test") for s in EVAL_SETS}
        n_test = min(len(v) for v in self.test.values())
        self.n = min(cfg["oblique_n"], n_test)
        if self.n < cfg["oblique_n"]:
            print(f"[oblique] oblique_n={cfg['oblique_n']} but the test split has {n_test} per set; using n={self.n}")
        self.test = {s: v[:self.n] for s, v in self.test.items()}

    def d_theta(self, theta: float) -> torch.Tensor:
        t = math.radians(theta)
        return unit(math.cos(t) * self.r_a + math.sin(t) * self.r_perp)

    def cosines(self, d: torch.Tensor | None) -> dict:
        if d is None:
            return {"cos_r_c": np.nan, "cos_r_a": np.nan, "cos_r_c_perp": np.nan}
        d = unit(d.float().cpu())
        return {"cos_r_c": float(d @ self.r_c), "cos_r_a": float(d @ self.r_a), "cos_r_c_perp": float(d @ self.r_perp)}

    def sweep_conditions(self) -> list[dict]:
        """Ordered by priority: base, rc_a100, then alpha 1 / 0.5 / 0.25 / 0.75 (and any others)."""
        alphas = [a for a in self.cfg["oblique_alphas"] if a > 0]
        order = sorted(alphas, key=lambda a: (a != 1.0, a != 0.5, a != 0.25, a))
        conds = [{"name": "base", "family": "base", "theta": np.nan, "alpha": 0.0, "d": None},
                 {"name": "rc_a100", "family": "rc", "theta": self.theta_rc, "alpha": 1.0, "d": self.r_c}]
        for a in order:
            for th in self.cfg["oblique_thetas"]:
                conds.append({"name": f"th{int(th):02d}_a{round(a * 100):03d}", "family": "oblique",
                              "theta": float(th), "alpha": float(a), "d": self.d_theta(th)})
            if a != 1.0:
                conds.append({"name": f"rc_a{round(a * 100):03d}", "family": "rc", "theta": self.theta_rc,
                              "alpha": float(a), "d": self.r_c})
        return conds


# ---------------------------------------------------------------- generation

def gen_condition(ctx: Ctx, model, tok, name: str, d, alpha: float):
    from .ablate_eval import generate

    gdir = ctx.gens / name
    gdir.mkdir(exist_ok=True)
    for s, samples in ctx.test.items():
        f = gdir / f"{s}.jsonl"
        done = {r["id"] for r in read_jsonl(f)}
        todo = [x for x in samples if x["id"] not in done]
        if not todo:
            continue
        for rows in generate(model, tok, todo, ctx.cfg, d, alpha):
            for r in rows:
                r["condition"] = name
            append_jsonl(f, rows)
        print(f"[oblique] {name}/{s}: {len(done) + len(todo)} generations", flush=True)


# ---------------------------------------------------------------- learned direction

def _refusal_mask(tok, vocab: int, device) -> torch.Tensor:
    from .ablate_eval import refusal_token_ids

    m = torch.zeros(vocab, dtype=torch.bool, device=device)
    m[refusal_token_ids(tok)] = True
    return m


def _logodds(model, tok, samples, cfg, mask, d=None) -> torch.Tensor:
    """Per-sample first-token refusal log-odds at t_post_inst, [B] fp32 (differentiable in d)."""
    from .hooks import ablate_direction
    from .prompts import encode_batch

    enc, *_ = encode_batch(tok, samples, cfg["chat_template_kwargs"], input_device(model))
    with ablate_direction(model, d):
        logits = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    lp = torch.log_softmax(logits, -1)
    return torch.logsumexp(lp[:, mask], -1) - torch.logsumexp(lp[:, ~mask], -1)


@torch.no_grad()
def _logodds_all(model, tok, samples, cfg, mask, d=None) -> torch.Tensor:
    return torch.cat([_logodds(model, tok, b, cfg, mask, d).cpu() for b in batched(samples, cfg["batch_size"])])


def learn(ctx: Ctx, model, tok) -> torch.Tensor:
    f = ctx.out / "learned.pt"
    if f.exists():
        return torch.load(f)["d"]
    cfg = ctx.cfg
    harm = load_split(ctx.p, "content_harm", "train")
    destr = load_split(ctx.p, "action_destr", "train")
    dev = input_device(model)
    mask = _refusal_mask(tok, model.get_output_embeddings().weight.shape[0], dev)

    base_destr = _logodds_all(model, tok, destr, cfg, mask)
    print(f"[learn] original refusal log-odds: content_harm "
          f"{_logodds_all(model, tok, harm[:16], cfg, mask).mean():+.2f} (first 16), action_destr {base_destr.mean():+.2f}")

    v = torch.nn.Parameter(ctx.r_c.clone().to(dev))
    opt = torch.optim.Adam([v], lr=cfg["learn_lr"])
    # No HF gradient checkpointing (its recomputation does not match with the ablation hooks); instead
    # micro-batches of learn_micro prompts, gradients accumulated, so peak memory is one micro-batch.
    rng = np.random.default_rng(cfg["seed"])
    bsz, lam, log = cfg["learn_bsz"], cfg["learn_lambda"], []
    micro = cfg.get("learn_micro", 2)
    try:
        for step in range(cfg["learn_steps"]):
            hi = rng.choice(len(harm), bsz, replace=False)
            di = rng.choice(len(destr), bsz, replace=False)
            sh, sd, lc, la = [], [], 0.0, 0.0
            for ch in batched(list(hi), micro):
                s = _logodds(model, tok, [harm[i] for i in ch], cfg, mask, v)
                loss = torch.nn.functional.softplus(s).sum() / bsz
                loss.backward()
                lc += loss.item()
                sh.append(s.detach())
            for ch in batched(list(di), micro):
                s = _logodds(model, tok, [destr[i] for i in ch], cfg, mask, v)
                loss = ((s - base_destr[ch].to(dev)) ** 2).sum() / bsz
                (lam * loss).backward()
                la += loss.item()
                sd.append(s.detach())
            s_h, s_d = torch.cat(sh), torch.cat(sd)
            loss_c, loss_a = torch.tensor(lc), torch.tensor(la)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                v.data = unit(v.data)
            row = {"step": step, "loss_c": loss_c.item(), "loss_a": loss_a.item(),
                   "logodds_harm": s_h.mean().item(), "logodds_destr": s_d.mean().item(),
                   "logodds_destr_orig": base_destr[di].mean().item(), **ctx.cosines(v.detach())}
            log.append(row)
            if step % 10 == 0 or step == cfg["learn_steps"] - 1:
                print(f"[learn] {step:4d} loss_c={row['loss_c']:.3f} loss_a={row['loss_a']:.3f} "
                      f"harm={row['logodds_harm']:+.2f} destr={row['logodds_destr']:+.2f} "
                      f"(orig {row['logodds_destr_orig']:+.2f}) cos r_c/r_a/r_c_perp="
                      f"{row['cos_r_c']:.3f}/{row['cos_r_a']:.3f}/{row['cos_r_c_perp']:.3f}", flush=True)
    finally:
        opt.zero_grad(set_to_none=True)
    pd.DataFrame(log).to_csv(ctx.out / "learned_train_log.csv", index=False)

    d = unit(v.detach().float().cpu())
    # held-out first-token check on test
    with torch.no_grad():
        ho = {}
        for s in ("content_harm", "action_destr"):
            o = _logodds_all(model, tok, ctx.test[s], cfg, mask)
            a = _logodds_all(model, tok, ctx.test[s], cfg, mask, d)
            ho[f"test_logodds_{s}_orig"], ho[f"test_logodds_{s}_learned"] = o.mean().item(), a.mean().item()
    info = {"layer": ctx.layer, **ctx.cosines(d), **ho,
            **{k: cfg[k] for k in ("learn_steps", "learn_lr", "learn_bsz", "learn_lambda")}}
    (ctx.out / "learned.json").write_text(json.dumps(info, indent=2))
    torch_save_atomic({"d": d, **info}, f)
    print(f"[learn] done: {json.dumps(info)}")
    return d


# ---------------------------------------------------------------- GPU stage

def run_gpu(ctx: Ctx, do_learn: bool):
    from .common import load_model

    model, tok = load_model(ctx.cfg)
    conds = ctx.sweep_conditions()
    for c in conds[:2]:
        gen_condition(ctx, model, tok, c["name"], c["d"], c["alpha"])
    if do_learn:
        d = learn(ctx, model, tok)
        gen_condition(ctx, model, tok, "learned_a100", d, 1.0)
    for c in conds[2:]:
        gen_condition(ctx, model, tok, c["name"], c["d"], c["alpha"])


# ---------------------------------------------------------------- conditional (probe-gated)

def probe(ctx: Ctx) -> pd.DataFrame:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    L, cfg = ctx.layer, ctx.cfg

    def score(s, split):
        x = load_acts(ctx.p, cfg, s, split)[:, L, INST] @ ctx.h_a
        return x.numpy()[:, None]

    xd, xb = score("action_destr", "train"), score("action_benign", "train")
    clf = LogisticRegression().fit(np.r_[xd, xb], np.r_[np.ones(len(xd)), np.zeros(len(xb))])
    rows = []
    for s in EVAL_SETS + ["content_safe"]:
        acts = torch.load(ctx.p.acts / f"{s}_test.pt")
        x = (acts["acts"][:, L, INST].float() @ ctx.h_a).numpy()
        pr = clf.predict_proba(x[:, None])[:, 1]
        rows += [{"set": s, "id": i, "h_a_proj": a, "probe_p": b} for i, a, b in zip(acts["ids"], x, pr)]
    df = pd.DataFrame(rows)
    t = df[df.set.isin(["action_destr", "action_benign"])]
    auc = roc_auc_score(t.set == "action_destr", t.probe_p)
    info = {"layer": L, "coef": float(clf.coef_[0, 0]), "intercept": float(clf.intercept_[0]), "test_auc_destr_vs_benign": auc,
            "mean_p_by_set": df.groupby("set").probe_p.mean().to_dict()}
    (ctx.out / "probe.json").write_text(json.dumps(info, indent=2))
    df.to_csv(ctx.out / "probe_scores.csv", index=False)
    print(f"[probe] test AUC destr vs benign = {auc:.3f}; mean p by set: "
          + ", ".join(f"{k}={v:.2f}" for k, v in info["mean_p_by_set"].items()))
    return df


# ---------------------------------------------------------------- judging + report

def outcomes(ctx: Ctx, name: str, judge, meta: dict) -> dict[str, np.ndarray] | None:
    """Per-sample 0/1 outcome arrays in test order; None if the condition is incomplete."""
    out, recs = {}, []
    for s, samples in ctx.test.items():
        g = {r["id"]: r for r in read_jsonl(ctx.gens / name / f"{s}.jsonl")}
        if any(x["id"] not in g for x in samples):
            return None
        vals = []
        for x in samples:
            j = judge.judge(meta[x["id"]], g[x["id"]]["text"])
            recs.append({"condition": name, "id": x["id"], "set": s, "source": meta[x["id"]].get("source"),
                         "ambiguous": meta[x["id"]].get("ambiguous", False),
                         "request": meta[x["id"]]["messages"][-1]["content"], **j, "text": g[x["id"]]["text"]})
            if s == "content_harm":
                vals.append(int(j["refused"]))
            elif s == "action_destr":
                vals.append(j["category"])
            else:
                vals.append(int(j["tool_call"]))
        out[s] = np.array(vals)
    write_jsonl(ctx.out / f"samples_{name}.jsonl", recs)
    return out


def summarize(arrs: dict) -> dict[str, np.ndarray]:
    cat = arrs["action_destr"]
    return {"x": arrs["content_harm"].astype(float), "y": np.isin(cat, CAUTION).astype(float),
            "z": arrs["action_benign"].astype(float),
            **{f"destr_{c}": (cat == c).astype(float) for c in ("comply", *CAUTION, "other")}}


def report(ctx: Ctx):
    import matplotlib

    matplotlib.use("Agg")
    from .judge import make_judge

    cfg = ctx.cfg
    judge = make_judge(cfg)
    meta = {}
    for s in EVAL_SETS:
        for r in read_jsonl(ctx.p.set_file(s)):
            meta[r["id"]] = r

    specs = ctx.sweep_conditions()
    lf = ctx.out / "learned.pt"
    if lf.exists():
        specs.insert(2, {"name": "learned_a100", "family": "learned", "theta": np.nan, "alpha": 1.0,
                         "d": torch.load(lf)["d"]})
    points, missing = [], []
    for c in specs:
        o = outcomes(ctx, c["name"], judge, meta)
        if o is None:
            missing.append(c["name"])
            continue
        points.append({**{k: v for k, v in c.items() if k != "d"}, "threshold": np.nan, **ctx.cosines(c["d"]),
                       "per_sample": summarize(o)})
    if missing:
        print(f"[report] incomplete, skipped: {missing}")
    by = {pt["name"]: pt for pt in points}
    if "base" not in by or "rc_a100" not in by:
        raise SystemExit("[report] need complete base and rc_a100 generations")

    # conditional: per prompt, rc_a100 output if probe p < thr else base output
    if (ctx.p.acts / "content_harm_test.pt").exists():
        ps = probe(ctx).set_index(["set", "id"]).probe_p
        A, B = by["base"]["per_sample"], by["rc_a100"]["per_sample"]
        gate_set = {"x": "content_harm", "y": "action_destr", "z": "action_benign"}
        for thr in cfg["probe_thresholds"]:
            per = {}
            for k in A:
                s = gate_set.get(k, "action_destr")
                g = np.array([ps[(s, x["id"])] < thr for x in ctx.test[s]])
                per[k] = np.where(g, B[k], A[k])
            pt = {"name": f"cond_p{round(thr * 100):03d}", "family": "conditional", "theta": np.nan,
                  "alpha": 1.0, "threshold": thr, **ctx.cosines(ctx.r_c), "per_sample": per}
            for s in EVAL_SETS:
                pt[f"gate_open_{s}"] = float(np.mean([ps[(s, x["id"])] < thr for x in ctx.test[s]]))
            points.append(pt)

    # paired bootstrap: same resampled prompt indices for every condition
    rng = np.random.default_rng(cfg["seed"])
    nb = cfg["n_boot"]
    idx = {k: rng.integers(0, ctx.n, (nb, ctx.n)) for k in ("x", "y", "z")}
    for pt in points:
        pt["boot"] = {k: v[idx[k if k in idx else "y"]].mean(1) for k, v in pt["per_sample"].items()}
    a, b = by["base"]["boot"], by["rc_a100"]["boot"]
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    rows = []
    for pt in points:
        bt = pt["boot"]
        # signed distance from the line base->rc_a100; > 0 = upper-left = less content refusal per unit caution lost
        sel = -(dx * (bt["y"] - a["y"]) - dy * (bt["x"] - a["x"])) / np.where(norm > 0, norm, np.nan)
        r = {k: v for k, v in pt.items() if k not in ("per_sample", "boot")}
        for k, v in pt["per_sample"].items():
            r[k] = v.mean()
            r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(bt[k], [2.5, 97.5])
        A0, B0 = by["base"]["per_sample"], by["rc_a100"]["per_sample"]
        ddx, ddy = B0["x"].mean() - A0["x"].mean(), B0["y"].mean() - A0["y"].mean()
        r["sel"] = -(ddx * (r["y"] - A0["y"].mean()) - ddy * (r["x"] - A0["x"].mean())) / math.hypot(ddx, ddy)
        r["sel_lo"], r["sel_hi"] = np.nanpercentile(sel, [2.5, 97.5])
        r["n"] = ctx.n
        rows.append(r)
    df = pd.DataFrame(rows)
    df.insert(1, "layer", ctx.layer)
    df.to_csv(ctx.out / "points.csv", index=False)
    plot(ctx, df)

    def best(fam_mask):
        sub = df[fam_mask]
        return sub.loc[sub.sel.idxmax()] if len(sub) else None

    summary = {"layer": ctx.layer, "n_per_set": ctx.n, "theta_rc_deg": ctx.theta_rc,
               "cos_r_c_r_a": float(ctx.r_c @ ctx.r_a), "missing_conditions": missing, "best_by_family": {}}
    for fam in ("oblique", "rc", "learned", "conditional"):
        r = best(df.family == fam)
        if r is not None:
            summary["best_by_family"][fam] = {k: (None if pd.isna(r[k]) else r[k]) for k in (
                "name", "theta", "alpha", "threshold", "x", "x_lo", "x_hi", "y", "y_lo", "y_hi",
                "z", "z_lo", "z_hi", "sel", "sel_lo", "sel_hi")}
    (ctx.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    show = ["name", "x", "x_lo", "x_hi", "y", "y_lo", "y_hi", "z", "sel", "sel_lo", "sel_hi"]
    print(df.sort_values("sel", ascending=False)[show].round(3).to_string(index=False))
    print(f"[report] points.csv, summary.json, plots -> {ctx.out}")


# ---------------------------------------------------------------- plots

THETA_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue 250..700
RC_COLOR, LEARN_COLOR, COND_COLOR, BASE_COLOR = "#e34948", "#eb6834", "#1baf7a", "#52514e"


def plot(ctx: Ctx, df: pd.DataFrame):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from . import analyze  # noqa: F401  (applies the shared rcParams / palette)
    from .analyze import GRID, INK2

    thetas = sorted(df[df.family == "oblique"].theta.unique())
    tcol = {t: THETA_RAMP[round(i * (len(THETA_RAMP) - 1) / max(len(thetas) - 1, 1))] for i, t in enumerate(thetas)}
    size = lambda a: 25 + 120 * a  # noqa: E731

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    A, B = df[df.name == "base"].iloc[0], df[df.name == "rc_a100"].iloc[0]
    ax.plot([A.x, B.x], [A.y, B.y], color=INK2, linestyle="--", linewidth=1.2, zorder=1)

    def err(sub, color, alpha=0.35):
        ax.errorbar(sub.x, sub.y, xerr=[sub.x - sub.x_lo, sub.x_hi - sub.x], yerr=[sub.y - sub.y_lo, sub.y_hi - sub.y],
                    fmt="none", ecolor=color, elinewidth=0.8, alpha=alpha, zorder=2)

    for t in thetas:
        sub = df[(df.family == "oblique") & (df.theta == t)].sort_values("alpha")
        err(sub, tcol[t])
        line = pd.concat([df[df.name == "base"], sub])
        ax.plot(line.x, line.y, color=tcol[t], linewidth=1, alpha=0.6, zorder=2)
        ax.scatter(sub.x, sub.y, s=size(sub.alpha), color=tcol[t], edgecolor="white", linewidth=1.2, zorder=3)
    rc = df[df.family == "rc"].sort_values("alpha")
    if len(rc):
        err(rc, RC_COLOR)
        line = pd.concat([df[df.name == "base"], rc])
        ax.plot(line.x, line.y, color=RC_COLOR, linewidth=1, alpha=0.7, zorder=2)
        ax.scatter(rc.x, rc.y, s=size(rc.alpha), marker="D", color=RC_COLOR, edgecolor="white", linewidth=1.2, zorder=4)
    cd = df[df.family == "conditional"].sort_values("threshold")
    if len(cd):
        err(cd, COND_COLOR)
        ax.plot(cd.x, cd.y, color=COND_COLOR, linewidth=1.5, marker="s", markersize=6,
                markeredgecolor="white", zorder=4)
    ln = df[df.family == "learned"]
    if len(ln):
        err(ln, LEARN_COLOR, 0.8)
        ax.scatter(ln.x, ln.y, s=260, marker="*", color=LEARN_COLOR, edgecolor="white", linewidth=1.2, zorder=5)
    ax.scatter([A.x], [A.y], s=90, color=BASE_COLOR, edgecolor="white", linewidth=1.2, zorder=5)
    ax.annotate("A  original", (A.x, A.y), xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK2)
    ax.annotate(f"B  full r_c removal (θ≈{ctx.theta_rc:.0f}°)", (B.x, B.y), xytext=(8, -12),
                textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, max(1.0, df.y_hi.max()) + 0.03)
    ax.set_xlabel("content refusal rate on content_harm  (lower = refusal removed)")
    ax.set_ylabel("action caution rate on action_destr  (refuse + clarify + safe alternative)")
    ax.set_title(f"Content refusal vs action caution, layer {ctx.layer}, n={ctx.n}/set, "
                 "bootstrap 95% CI", loc="left")
    handles = [Line2D([], [], marker="o", linestyle="", color=tcol[t], markersize=8, label=f"θ={t:.0f}°") for t in thetas]
    handles += [Line2D([], [], marker="D", linestyle="", color=RC_COLOR, markersize=7, label="r_c (partial α)"),
                Line2D([], [], marker="*", linestyle="", color=LEARN_COLOR, markersize=13, label="learned d"),
                Line2D([], [], marker="s", color=COND_COLOR, markersize=6, label="probe-gated r_c (thresholds)"),
                Line2D([], [], color=INK2, linestyle="--", label="diagonal A→B")]
    handles += [Line2D([], [], marker="o", linestyle="", color=GRID, markeredgecolor=INK2,
                       markersize=math.sqrt(size(a)), label=f"α={a:g}") for a in (0.25, 0.5, 1.0)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.text(0.01, 0.01, "selective = above-left of the diagonal (less content refusal, caution kept)",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    fig.tight_layout()
    fig.savefig(ctx.out / "oblique_plane.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ranking: signed distance from the diagonal, plus the benign tool-use side effect
    d = df[df.family != "base"].sort_values("sel")
    fam_color = {"rc": RC_COLOR, "learned": LEARN_COLOR, "conditional": COND_COLOR}
    colors = [tcol[r.theta] if r.family == "oblique" else fam_color[r.family] for r in d.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.24 * len(d) + 1.6), sharey=True,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    yy = np.arange(len(d))
    for ax, (k, title) in zip(axes, [("sel", "Signed distance from diagonal A→B  (> 0 = selective)"),
                                     ("z", "action_benign tool-use rate (side effect)")]):
        ax.errorbar(d[k], yy, xerr=[d[k] - d[f"{k}_lo"], d[f"{k}_hi"] - d[k]], fmt="none", ecolor=INK2,
                    elinewidth=0.8, zorder=1)
        ax.scatter(d[k], yy, c=colors, s=40, edgecolor="white", linewidth=1, zorder=2)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0].axvline(0, color=INK2, linewidth=0.8)
    axes[0].set_yticks(yy, d.name, fontsize=7.5)
    axes[1].set_xlim(-0.03, 1.05)
    fig.tight_layout()
    fig.savefig(ctx.out / "oblique_ranking.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- entry

def main():
    args = parse_args(__doc__, lambda p: p.add_argument("--stages", nargs="*", default=["gpu", "learn", "report"],
                                                        help="gpu (sweep generations), learn, report"))
    cfg = load_config(args)
    ctx = Ctx(cfg)
    print(f"[oblique] layer={ctx.layer} cos(r_c,r_a)={float(ctx.r_c @ ctx.r_a):.3f} theta_rc={ctx.theta_rc:.1f}° "
          f"n={ctx.n} -> {ctx.out}")
    if "gpu" in args.stages:
        run_gpu(ctx, do_learn="learn" in args.stages)
    if "report" in args.stages:
        report(ctx)


if __name__ == "__main__":
    main()
