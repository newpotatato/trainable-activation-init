"""Is the 5-epoch pilot on average at least as good as the accepted norm (default init)?

Compares pilot5 with (a) the default init at the same training budget (30 epochs)
and (b) the default init trained 25 epochs longer (equal total compute), on all
eight tasks of the 3 x 128 MLP. Writes results/summary_vs_norm.md and
results/figures/fig8_vs_norm.png.

Per pair (task, seed): rel = (err_ref - err_pilot5) / err_ref, err = RMSE or 1 - accuracy;
positive = pilot5 better. A task's effect is the mean over its seeds; a function's
effect is the mean over tasks (each task weighs equally), with a 95 % bootstrap CI
(seeds resampled within each task).
"""
import os
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from analyze import ACT_NAMES, FIG_DIR, HERE, load_runs, per_run_metrics

RESULTS = os.path.join(HERE, "results")
FILES = ["runs.jsonl", "runs_pilot5.jsonl", "runs_pilot5b.jsonl", "runs_default_long.jsonl",
         "runs_default_long_smooth.jsonl", "runs_default_long_cw.jsonl"]
TASKS = ["california", "wine", "covertype", "fashion", "friedman", "friedman2", "friedman3", "sines"]
TASK_NAMES = {"california": "California", "wine": "Wine", "covertype": "Covertype", "fashion": "Fashion-MNIST",
              "friedman": "Friedman #1", "friedman2": "Friedman #2", "friedman3": "Friedman #3", "sines": "Синусоиды"}
ACTS = ["prelu", "swish", "aconc", "siglin"]
REFS = [("default", "стандартная, 30 эпох"), ("default_long", "стандартная, 55 эпох (равные затраты)")]
OUT = os.path.join(RESULTS, "summary_vs_norm.md")


def load():
    runs = []
    for f in FILES:
        path = os.path.join(RESULTS, f)
        if os.path.exists(path):
            runs += [r for r in load_runs(path) if r.get("arch", "mlp3") == "mlp3"]
    per_run_metrics(runs)
    g = defaultdict(dict)
    for r in runs:
        err = r["m"]["test"] if r["task"] == "reg" else 1 - r["m"]["test"]
        g[(r["dataset"], r["act"], r["strategy"])][r["seed"]] = err
    return g


def paired(g, task, act, ref):
    a, b = g.get((task, act, "pilot5"), {}), g.get((task, act, ref), {})
    seeds = sorted(set(a) & set(b))
    return np.array([(b[s] - a[s]) / b[s] for s in seeds])


def holm(p):
    order = np.argsort(p)
    adj, prev = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = prev
    return adj


def summarise(g, ref, rng):
    rows = []
    for act in ACTS:
        per_task = {t: paired(g, t, act, ref) for t in TASKS}
        per_task = {t: v for t, v in per_task.items() if len(v)}
        if not per_task:
            continue
        means = {t: v.mean() for t, v in per_task.items()}
        boot = np.mean([[rng.choice(v, len(v)).mean() for v in per_task.values()] for _ in range(5000)], axis=1)
        allrel = np.concatenate(list(per_task.values()))
        wins, n = int((allrel > 0).sum()), int((allrel != 0).sum())
        worst = min(means, key=means.get)
        rows.append({"act": act, "tasks": len(per_task), "mean": 100 * np.mean(list(means.values())),
                     "median": 100 * np.median(list(means.values())),
                     "not_worse": sum(v >= 0 for v in means.values()),
                     "lo": 100 * np.percentile(boot, 2.5), "hi": 100 * np.percentile(boot, 97.5),
                     "wins": wins, "n": n, "p": binomtest(wins, n).pvalue if n else 1.0,
                     "worst": worst, "worst_val": 100 * means[worst], "per_task": means})
    adj = holm(np.array([r["p"] for r in rows]))
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    return rows


def fmt_p(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def num(x, d=1):
    return f"{x:+.{d}f}".replace(".", ",")


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    g = load()
    lines = ["# Пилот (5 эпох) против принятой нормы (стандартная инициализация)", "",
             "Сеть 3 × 128, все 8 задач. Положительное значение — пилот лучше. Эффект задачи — среднее по seed; "
             "эффект функции — среднее по задачам с 95 % бутстреп-интервалом. Пары с ничьей (одинаковый выбор) "
             "в знаковом тесте не учитываются.", ""]
    results = {}
    for ref, ref_name in REFS:
        rows = summarise(g, ref, rng)
        results[ref] = rows
        lines += [f"## Против: {ref_name}", "",
                  "| Функция | Средний выигрыш, % [95 % ДИ] | Медиана по задачам, % | Задач, где не хуже | "
                  "Выиграно пар | p (Холм) | Худшая задача, % |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {ACT_NAMES[r['act']]} | {num(r['mean'])} [{num(r['lo'])}; {num(r['hi'])}] | "
                         f"{num(r['median'], 2)} | {r['not_worse']}/{r['tasks']} | "
                         f"{r['wins']}/{r['n']} | {fmt_p(r['p_holm'])} | {TASK_NAMES[r['worst']]} ({num(r['worst_val'])}) |")
        lines += ["", "По задачам, %:", "",
                  "| Функция | " + " | ".join(TASK_NAMES[t] for t in TASKS) + " |", "|" + "---|" * (len(TASKS) + 1)]
        for r in rows:
            lines.append(f"| {ACT_NAMES[r['act']]} | " + " | ".join(
                num(100 * r["per_task"][t]) if t in r["per_task"] else "—" for t in TASKS) + " |")
        lines.append("")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    plot(results)


def plot(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(REFS), figsize=(6.2 * len(REFS), 3.8), sharey=True)
    markers = dict(zip(TASKS, "o s ^ D v P X *".split()))
    for ax, (ref, ref_name) in zip(np.atleast_1d(axes), REFS):
        rows = results.get(ref, [])
        for i, r in enumerate(rows):
            for t, v in r["per_task"].items():
                ax.scatter(i + (TASKS.index(t) - 3.5) * 0.05, np.clip(100 * v, -12, 12), marker=markers[t],
                           s=22, color="#9E9E9E", label=TASK_NAMES[t] if i == 0 else None, zorder=2)
            ax.errorbar(i + 0.32, np.clip(r["mean"], -12, 12), yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                        fmt="o", color="#7B4FA0", capsize=4, ms=7, zorder=3,
                        label="среднее по задачам, 95 % ДИ" if i == 0 else None)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([ACT_NAMES[r["act"]].replace(" (наша)", "\n(наша)") for r in rows], fontsize=9)
        ax.set_ylim(-12, 12)
        ax.set_title(f"против: {ref_name}", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("выигрыш пилота (5 эпох), %\n(значения за ±12 % прижаты к краю)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(os.path.join(FIG_DIR, "fig8_vs_norm.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
