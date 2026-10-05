"""Article figures and compact tables (results/figures/fig*.png, results/article_tables.md).

Improvement of a strategy over the default init is a relative error reduction,
so regression and classification share one axis (positive = better):
  regression:     (RMSE_default - RMSE_strategy) / RMSE_default
  classification: (err_default - err_strategy) / err_default, err = 1 - accuracy
"""
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from analyze import ACT_NAMES, FIG_DIR, HERE, RUNS_MLP8, load_runs, per_run_metrics

DS_ORDER = ["california", "wine", "covertype", "fashion"]
DS_NAMES = {"california": "California Housing", "wine": "Wine Quality",
            "covertype": "Covertype", "fashion": "Fashion-MNIST"}
ACTS = ["prelu", "swish", "aconc", "siglin"]
STRATS = {"random": ("случайная", "#9E9E9E"), "pilot": ("пробные прогоны", "#E07B39"),
          "transfer": ("перенос с других датасетов", "#3A923A")}
DEFAULT_COLOR = "#3B6FB6"
TABLES = os.path.join(HERE, "results", "article_tables.md")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def error(r):
    """Lower-is-better error used for relative improvement."""
    return r["m"]["test"] if r["task"] == "reg" else 1.0 - r["m"]["test"]


def fig_shape():
    x = np.linspace(-4, 4, 400)
    sig = lambda z: 1 / (1 + np.exp(-z))
    variants = [((3.6, 1.0, 0.1), "B=3.6, n=1, L=0.1 (стандартная)", DEFAULT_COLOR, "-"),
                ((1.0, 1.0, 0.05), "B=1, n=1, L=0.05", "#E07B39", "--"),
                ((4.0, 1.0, 0.2), "B=4, n=1, L=0.2", "#3A923A", "-.")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for (B, n, L), label, c, ls in variants:
        s = sig(B * x)
        a1.plot(x, n * s + L * x, color=c, ls=ls, label=label)
        a2.plot(x, n * B * s * (1 - s) + L, color=c, ls=ls)
        a2.axhline(L, color=c, lw=0.6, ls=":")
    a1.plot(x, sig(x), color="black", lw=1, ls=":", label="сигмоида σ(x)")
    a2.plot(x, sig(x) * (1 - sig(x)), color="black", lw=1, ls=":")
    a1.set_title("f(x) = n·σ(Bx) + L·x")
    a2.set_title("f′(x) ∈ [L, L + nB/4]  (пунктир — «пол» L)")
    for a in (a1, a2):
        a.set_xlabel("x")
        a.axhline(0, color="grey", lw=0.5)
    a1.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_siglin_shape.png"), dpi=200)
    plt.close(fig)


def fig_improvement(groups, datasets):
    # random init is reported in the tables; its large losses would hide the signal here
    strategies = [s for s in ("pilot", "transfer") if any(k[2] == s for k in groups)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.6), sharey=True)
    width = 0.8 / len(strategies)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        for j, st in enumerate(strategies):
            means, sds = [], []
            for act in ACTS:
                base = {r["seed"]: error(r) for r in groups.get((ds, act, "default"), [])}
                other = {r["seed"]: error(r) for r in groups.get((ds, act, st), [])}
                seeds = sorted(set(base) & set(other))
                rel = [100 * (base[s] - other[s]) / base[s] for s in seeds] or [np.nan]
                means.append(np.mean(rel))
                sds.append(np.std(rel, ddof=1) if len(rel) > 1 else 0)
            xs = np.arange(len(ACTS)) + (j - (len(strategies) - 1) / 2) * width
            ax.bar(xs, means, width, yerr=sds, color=STRATS[st][1], label=STRATS[st][0], capsize=2, error_kw={"lw": 0.8})
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(ACTS)))
        ax.set_xticklabels([ACT_NAMES[a].replace(" (наша)", "\n(наша)") for a in ACTS], fontsize=8)
        ax.set_title(DS_NAMES[ds])
    np.atleast_1d(axes)[0].set_ylabel("снижение ошибки на test\nотносительно стандартной, %")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(strategies), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG_DIR, "fig2_improvement.png"), dpi=200)
    plt.close(fig)


def fig_drift(runs):
    panels = [("prelu", "a", "PReLU: a"), ("swish", "beta", "Swish-β: β"), ("aconc", "beta", "ACON-C: β"),
              ("siglin", "B", "SigLin-T: B"), ("siglin", "L", "SigLin-T: L")]
    colors = {"default": DEFAULT_COLOR, **{k: v[1] for k, v in STRATS.items()}}
    names = {"default": "стандартная", **{k: v[0] for k, v in STRATS.items()}}
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.1))
    for ax, (act, p, title) in zip(axes, panels):
        lo, hi = np.inf, -np.inf
        for st in colors:
            xs, ys = [], []
            for r in runs:
                if r["act"] != act or r["strategy"] != st or not isinstance(r["init"].get(p), (int, float)):
                    continue
                for layer in r["final_params"]:
                    xs.append(r["init"][p])
                    ys.append(layer[p])
            if xs:
                jitter = np.random.default_rng(0).normal(0, 0.01 * (max(xs) - min(xs) + 1e-3), len(xs))
                ax.scatter(np.array(xs) + jitter, ys, s=7, alpha=0.5, color=colors[st], label=names[st])
                lo, hi = min(lo, min(xs), min(ys)), max(hi, max(xs), max(ys))
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("начальное значение")
    axes[0].set_ylabel("после обучения\n(среднее по нейронам слоя)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03), markerscale=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG_DIR, "fig3_param_drift.png"), dpi=200)
    plt.close(fig)


def fig_curves(groups, datasets):
    acts = ["swish", "siglin"]
    fig, axes = plt.subplots(len(acts), len(datasets), figsize=(3.3 * len(datasets), 2.7 * len(acts)), sharex=True)
    for i, act in enumerate(acts):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            for st, color in [("default", DEFAULT_COLOR), ("pilot", STRATS["pilot"][1]), ("transfer", STRATS["transfer"][1])]:
                g = groups.get((ds, act, st), [])
                if not g:
                    continue
                c = np.array([[e["val"]["loss"] for e in r["curve"]] for r in g])
                x = np.arange(1, c.shape[1] + 1)
                ax.plot(x, c.mean(0), color=color, label={"default": "стандартная", **{k: v[0] for k, v in STRATS.items()}}[st])
                ax.fill_between(x, c.mean(0) - c.std(0), c.mean(0) + c.std(0), color=color, alpha=0.15)
            g = groups.get((ds, "relu", "default"), [])
            if g:
                c = np.array([[e["val"]["loss"] for e in r["curve"]] for r in g])
                ax.plot(np.arange(1, c.shape[1] + 1), c.mean(0), color="black", ls=":", lw=1, label="ReLU")
            lo = min(min(e["val"]["loss"] for r in groups.get((ds, act, st), []) for e in r["curve"])
                     for st in ("default", "pilot", "transfer") if groups.get((ds, act, st)))
            ax.set_ylim(lo * 0.97, lo * 1.5)
            if i == 0:
                ax.set_title(DS_NAMES[ds])
            if j == 0:
                ax.set_ylabel(f"{ACT_NAMES[act]}\nval loss")
            if i == len(acts) - 1:
                ax.set_xlabel("эпоха")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG_DIR, "fig4_curves.png"), dpi=200)
    plt.close(fig)


def sign_counts(groups, datasets, acts, st):
    """Wins of strategy st over the default init across all (dataset, act, seed) pairs."""
    w = n = 0
    for ds in datasets:
        for act in acts:
            base = {r["seed"]: error(r) for r in groups.get((ds, act, "default"), [])}
            other = {r["seed"]: error(r) for r in groups.get((ds, act, st), [])}
            for s in set(base) & set(other):
                n += 1
                w += other[s] < base[s]
    return w, n


def tables(groups, datasets):
    strategies = ["default", "pilot", "transfer"]
    names = {"default": "станд.", "pilot": "пилот", "transfer": "перенос"}
    lines = ["# Таблицы для статьи", "",
             "Test-метрика (RMSE для регрессии — меньше лучше; accuracy для классификации — больше лучше), среднее ± std по 5 seed.", ""]
    header = "| Функция | Инициализация | " + " | ".join(DS_NAMES[d] for d in datasets) + " |"
    lines += [header, "|" + "---|" * (2 + len(datasets))]
    for act in ["relu", "gelu"]:
        cells = []
        for ds in datasets:
            v = [r["m"]["test"] for r in groups.get((ds, act, "default"), [])]
            cells.append(f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" if v else "—")
        lines.append(f"| {ACT_NAMES[act]} | — | " + " | ".join(cells) + " |")
    for act in ACTS:
        for st in strategies:
            cells = []
            for ds in datasets:
                v = [r["m"]["test"] for r in groups.get((ds, act, st), [])]
                cells.append(f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" if v else "—")
            lines.append(f"| {ACT_NAMES[act]} | {names[st]} | " + " | ".join(cells) + " |")
    lines += ["", "Число seed (из 5), на которых стратегия лучше стандартной инициализации по test-метрике:", ""]
    lines += ["| Функция | Стратегия | " + " | ".join(DS_NAMES[d] for d in datasets) + " |", "|" + "---|" * (2 + len(datasets))]
    for act in ACTS:
        for st in ["random", "pilot", "transfer"]:
            cells = []
            for ds in datasets:
                base = {r["seed"]: error(r) for r in groups.get((ds, act, "default"), [])}
                other = {r["seed"]: error(r) for r in groups.get((ds, act, st), [])}
                seeds = sorted(set(base) & set(other))
                cells.append(f"{sum(other[s] < base[s] for s in seeds)}/{len(seeds)}" if seeds else "—")
            lines.append(f"| {ACT_NAMES[act]} | {STRATS[st][0]} | " + " | ".join(cells) + " |")
    lines += ["", "Знаковый тест по всем парам (набор данных × seed): сколько раз стратегия лучше стандартной "
                  "инициализации, и двусторонний p-value точного биномиального теста (H0: вероятность выигрыша 0,5).", ""]
    lines += ["| Функция | Стратегия | Выигрышей | p-value |", "|---|---|---|---|"]
    for act in ACTS + ["all"]:
        for st in ["random", "pilot", "transfer"]:
            w, n = sign_counts(groups, datasets, ACTS if act == "all" else [act], st)
            if n:
                name = "все четыре" if act == "all" else ACT_NAMES[act]
                lines.append(f"| {name} | {STRATS[st][0]} | {w}/{n} | {binomtest(w, n).pvalue:.3g} |")
    lines += ["", "Дополнительные эпохи на пробные прогоны: 4 кандидата × 2 эпохи = 8 (при основном обучении 30 эпох).", ""]
    with open(TABLES, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def fig_deep(groups8, datasets):
    """Deep network (8 x 128): val loss curves of Sigmoid vs SigLin-T vs ReLU."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 2.9))
    series = [("sigmoid", "default", "Sigmoid", "#9E9E9E", "-"),
              ("siglin", "default", "SigLin-T (стандартная)", DEFAULT_COLOR, "-"),
              ("siglin", "transfer", "SigLin-T (перенос)", STRATS["transfer"][1], "-"),
              ("relu", "default", "ReLU", "black", ":")]
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        lo, hi = np.inf, -np.inf
        for act, st, label, color, ls in series:
            g = groups8.get((ds, act, st), [])
            if not g:
                continue
            c = np.array([[e["val"]["loss"] for e in r["curve"]] for r in g])
            x = np.arange(1, c.shape[1] + 1)
            ax.plot(x, c.mean(0), color=color, ls=ls, label=label)
            ax.fill_between(x, c.mean(0) - c.std(0), c.mean(0) + c.std(0), color=color, alpha=0.12)
            lo = min(lo, c.mean(0).min())
            hi = max(hi, c.mean(0)[2:].max())
        ax.set_ylim(lo * 0.95, hi * 1.05)
        ax.set_title(DS_NAMES[ds])
        ax.set_xlabel("эпоха")
    np.atleast_1d(axes)[0].set_ylabel("val loss")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(FIG_DIR, "fig5_deep.png"), dpi=200)
    plt.close(fig)


SMOOTH = ["friedman", "friedman2", "friedman3", "sines"]
SMOOTH_NAMES = {"friedman": "Friedman #1", "friedman2": "Friedman #2", "friedman3": "Friedman #3", "sines": "Синусоиды"}
RESULTS_DIR = os.path.join(HERE, "results")


def _load_groups(files):
    runs = []
    for f in files:
        path = os.path.join(RESULTS_DIR, f)
        if os.path.exists(path):
            runs += load_runs(path)
    per_run_metrics(runs)
    g = defaultdict(list)
    for r in runs:
        g[(r["dataset"], r["act"], r["strategy"])].append(r)
    return g


def fig_smooth():
    """Smooth regressions: how far each strategy is from the better of the two fixed inits
    (default or transferred) on every task; positive = worse than the best fixed choice."""
    g = _load_groups(["runs.jsonl", "runs_pilot5.jsonl"])
    datasets = [d for d in SMOOTH if g.get((d, "swish", "pilot5"))]
    if not datasets:
        return
    strategies = [("default", "стандартная", DEFAULT_COLOR), ("transfer", "перенос", STRATS["transfer"][1]),
                  ("pilot5", "пилот, 5 эпох", "#7B4FA0")]
    cap = 8.0
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.3 * len(datasets), 3.6), sharey=True)
    width = 0.8 / len(strategies)
    for ax, ds in zip(np.atleast_1d(axes), datasets):
        for j, (st, label, color) in enumerate(strategies):
            vals = []
            for act in ACTS:
                mean = lambda s_: np.mean([r["m"]["test"] for r in g[(ds, act, s_)]])
                best = min(mean("default"), mean("transfer"))
                vals.append(100 * (mean(st) - best) / best)
            xs = np.arange(len(ACTS)) + (j - (len(strategies) - 1) / 2) * width
            ax.bar(xs, np.minimum(vals, cap), width, color=color, label=label)
            for x, v in zip(xs, vals):
                if v > cap:
                    ax.text(x, cap * 0.97, f"{v:.0f}%", rotation=90, ha="center", va="top", fontsize=7, color="white")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylim(-1, cap)
        ax.set_xticks(range(len(ACTS)))
        ax.set_xticklabels([ACT_NAMES[a].replace(" (наша)", "\n(наша)") for a in ACTS], fontsize=8)
        ax.set_title(SMOOTH_NAMES[ds])
    np.atleast_1d(axes)[0].set_ylabel("отставание RMSE от лучшего\nфиксированного варианта, %")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(FIG_DIR, "fig6_smooth.png"), dpi=200)
    plt.close(fig)


def fig_sweep():
    """Friedman #1: test RMSE vs pilot length, with default and transfer as reference lines."""
    g = _load_groups(["runs.jsonl", "runs_pilot5.jsonl", "runs_pilot_sweep.jsonl"])
    lengths = [(1, "pilot_e1"), (2, "pilot_e2"), (3, "pilot_e3"), (5, "pilot5"), (8, "pilot_e8")]
    acts = [a for a in ("swish", "aconc") if g.get(("friedman", a, "pilot_e1"))]
    if not acts:
        return
    fig, axes = plt.subplots(1, len(acts), figsize=(4.6 * len(acts), 3.3))
    for ax, act in zip(np.atleast_1d(axes), acts):
        xs, mu, sd = [], [], []
        for t, st in lengths:
            v = [r["m"]["test"] for r in g.get(("friedman", act, st), [])]
            if v:
                xs.append(t)
                mu.append(np.mean(v))
                sd.append(np.std(v, ddof=1))
        ax.errorbar(xs, mu, yerr=sd, color="#7B4FA0", marker="o", capsize=3, label="пилот (кандидаты + стандартная)")
        for st, label, color, ls in (("default", "стандартная", DEFAULT_COLOR, "--"),
                                     ("transfer", "перенос", STRATS["transfer"][1], ":")):
            v = [r["m"]["test"] for r in g.get(("friedman", act, st), [])]
            if v:
                ax.axhline(np.mean(v), color=color, ls=ls, label=label)
        ax.set_xticks([t for t, _ in lengths])
        ax.set_xlabel("длина пробного прогона, эпох")
        ax.set_title(f"Friedman #1, {ACT_NAMES[act]}")
    np.atleast_1d(axes)[0].set_ylabel("RMSE на test")
    np.atleast_1d(axes)[-1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_pilot_length.png"), dpi=200)
    plt.close(fig)


def main():
    runs = load_runs()
    per_run_metrics(runs)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["act"], r["strategy"])].append(r)
    datasets = [d for d in DS_ORDER if any(r["dataset"] == d for r in runs)]
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_shape()
    fig_improvement(groups, datasets)
    fig_drift([r for r in runs if r["dataset"] in DS_ORDER])
    fig_curves(groups, datasets)
    tables(groups, datasets)
    if os.path.exists(RUNS_MLP8):
        runs8 = load_runs(RUNS_MLP8)
        per_run_metrics(runs8)
        groups8 = defaultdict(list)
        for r in runs8:
            groups8[(r["dataset"], r["act"], r["strategy"])].append(r)
        fig_deep(groups8, [d for d in DS_ORDER if any(r["dataset"] == d for r in runs8)])
    fig_smooth()
    fig_sweep()
    print("figures:", sorted(os.listdir(FIG_DIR)))
    print("tables:", TABLES)


if __name__ == "__main__":
    main()
