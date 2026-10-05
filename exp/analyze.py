"""Summarise results/runs.jsonl into results/summary.md and plots.

Per run:
  test      -- test metric at the epoch with the lowest val loss (RMSE for
               regression in target-std units, ROC AUC for classification)
  ep_target -- first epoch whose val loss <= target; target per dataset is
               1.05 x the median (over all runs) of the best val loss.
               Runs that never reach it get EPOCHS + 1.
  ep_cost   -- ep_target + pilot epochs (fair compute accounting for "pilot")
  val@1, val@3 -- val loss after epochs 1 and 3
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs.jsonl")
OUT = os.path.join(HERE, "results", "summary.md")
FIG_DIR = os.path.join(HERE, "results", "figures")

ACT_ORDER = ["relu", "gelu", "prelu", "swish", "aconc", "siglin"]
STRAT_ORDER = ["default", "random", "pilot", "transfer"]
ACT_NAMES = {"relu": "ReLU", "gelu": "GELU", "sigmoid": "Sigmoid", "prelu": "PReLU", "swish": "Swish-β",
             "aconc": "ACON-C", "siglin": "SigLin-T (наша)"}
STRAT_NAMES = {"pilot_d": "пробные прогоны (+стандартная)", "default": "стандартная", "random": "случайная", "pilot": "пробные прогоны", "transfer": "перенос с других датасетов"}


RUNS_MLP8 = os.path.join(HERE, "results", "runs_mlp8.jsonl")


def load_runs(path=RUNS):
    runs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                runs.append(r)
    return runs


def per_run_metrics(runs):
    # the target is fixed by the main strategies so that add-on experiments (pilot_d) do not move it
    by_ds = defaultdict(list)
    for r in runs:
        if r["strategy"] != "pilot_d":
            by_ds[r["dataset"]].append(r)
    targets = {}
    for ds, rs in by_ds.items():
        targets[ds] = 1.05 * float(np.median([min(e["val"]["loss"] for e in r["curve"]) for r in rs]))
    for r in runs:
        curve = r["curve"]
        val = [e["val"]["loss"] for e in curve]
        best = int(np.argmin(val))
        key = "rmse" if r["task"] == "reg" else "acc"
        hit = [i + 1 for i, v in enumerate(val) if v <= targets[r["dataset"]]]
        ep_target = hit[0] if hit else len(curve) + 1
        extra = r["pilot"]["extra_epochs"] if r["pilot"] else 0
        r["m"] = {
            "test": curve[best]["test"][key],
            "ep_target": ep_target,
            "reached": bool(hit),
            "ep_cost": ep_target + extra,
            "val1": val[0],
            "val3": val[2],
        }
    return targets


def fmt(xs, digits=3):
    xs = np.asarray(xs, dtype=float)
    return f"{xs.mean():.{digits}f} ± {xs.std(ddof=1) if len(xs) > 1 else 0:.{digits}f}"


def main():
    runs = load_runs()
    targets = per_run_metrics(runs)
    groups = defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["act"], r["strategy"])].append(r)
    datasets = sorted({r["dataset"] for r in runs})
    lines = ["# Результаты: инициализация параметров обучаемых активаций", ""]
    lines.append(f"Всего успешных прогонов: {len(runs)}. Среднее ± std по seed.")
    lines.append("")
    lines.append("- **test**: RMSE (в единицах std цели, меньше — лучше) для регрессии, accuracy (больше — лучше) для классификации; "
                 "берётся в эпоху с минимальным val loss.")
    lines.append("- **эпох до цели**: первая эпоха, где val loss ≤ цели (1.05 × медиана лучших val loss по датасету); "
                 "не достигшие цели получают 31.")
    lines.append("- **с учётом пилота**: эпохи до цели + эпохи пробных прогонов (4 кандидата × 2 эпохи = 8).")
    lines.append("")

    for ds in datasets:
        task = next(r["task"] for r in runs if r["dataset"] == ds)
        lines.append(f"## {ds} ({'регрессия, RMSE' if task == 'reg' else 'классификация, accuracy'}; цель val loss = {targets[ds]:.4f})")
        lines.append("")
        lines.append("| Активация | Инициализация | n | test | эпох до цели | с учётом пилота | достигли цели | val loss после 1 эп. | после 3 эп. |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for act in ACT_ORDER:
            for st in STRAT_ORDER:
                g = groups.get((ds, act, st))
                if not g:
                    continue
                m = [r["m"] for r in g]
                lines.append(
                    f"| {ACT_NAMES[act]} | {STRAT_NAMES[st]} | {len(g)} | {fmt([x['test'] for x in m], 4)} | "
                    f"{fmt([x['ep_target'] for x in m], 1)} | {fmt([x['ep_cost'] for x in m], 1)} | "
                    f"{sum(x['reached'] for x in m)}/{len(m)} | {fmt([x['val1'] for x in m], 4)} | {fmt([x['val3'] for x in m], 4)} |"
                )
        lines.append("")

    # Paired comparison pilot vs default and random vs default on identical weight seeds
    lines.append("## Попарное сравнение с стандартной инициализацией (одинаковые seed весов)")
    lines.append("")
    lines.append("Δ = стратегия − стандартная. Для RMSE и эпох отрицательное Δ — лучше, для accuracy — положительное. "
                 "«Лучше» — число seed, где стратегия выиграла.")
    lines.append("")
    lines.append("| Датасет | Активация | Стратегия | Δ test | лучше по test | Δ эпох до цели | Δ эпох с учётом пилота | Δ val loss после 1 эп. |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in datasets:
        task = next(r["task"] for r in runs if r["dataset"] == ds)
        sign = -1 if task == "reg" else 1
        for act in ACT_ORDER[2:]:
            base = {r["seed"]: r["m"] for r in groups.get((ds, act, "default"), [])}
            for st in ("random", "pilot", "transfer"):
                other = {r["seed"]: r["m"] for r in groups.get((ds, act, st), [])}
                seeds = sorted(set(base) & set(other))
                if not seeds:
                    continue
                dt = [other[s]["test"] - base[s]["test"] for s in seeds]
                wins = sum(sign * d > 0 for d in dt)
                de = [other[s]["ep_target"] - base[s]["ep_target"] for s in seeds]
                dc = [other[s]["ep_cost"] - base[s]["ep_target"] for s in seeds]
                d1 = [other[s]["val1"] - base[s]["val1"] for s in seeds]
                lines.append(f"| {ds} | {ACT_NAMES[act]} | {STRAT_NAMES[st]} | {fmt(dt, 4)} | {wins}/{len(seeds)} | "
                             f"{fmt(de, 1)} | {fmt(dc, 1)} | {fmt(d1, 4)} |")
    lines.append("")

    # Which candidates the pilot procedure picked, and where the parameters ended up
    lines.append("## Что выбирает процедура пробных прогонов")
    lines.append("")
    for ds in datasets:
        for act in ACT_ORDER[2:]:
            g = groups.get((ds, act, "pilot"), [])
            if not g:
                continue
            picks = defaultdict(int)
            for r in g:
                picks[json.dumps(r["init"], sort_keys=True)] += 1
            picked = "; ".join(f"{k} ×{v}" for k, v in sorted(picks.items(), key=lambda kv: -kv[1]))
            lines.append(f"- **{ds} / {ACT_NAMES[act]}**: {picked}")
    lines.append("")
    lines.append("## Значения параметров активации после обучения (среднее по нейронам, слоям и seed)")
    lines.append("")
    for ds in datasets:
        for act in ACT_ORDER[2:]:
            parts = []
            for st in STRAT_ORDER:
                g = groups.get((ds, act, st), [])
                if not g:
                    continue
                vals = defaultdict(list)
                for r in g:
                    for layer in r["final_params"]:
                        for k, v in layer.items():
                            vals[k].append(v)
                parts.append(f"{STRAT_NAMES[st]}: " + ", ".join(f"{k}={np.mean(v):.3f}" for k, v in vals.items()))
            if parts:
                lines.append(f"- **{ds} / {ACT_NAMES[act]}** — " + "; ".join(parts))
    lines.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")
    plot(runs, groups, datasets)


def plot(runs, groups, datasets):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping figures")
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    colors = {"default": "#4C72B0", "random": "#999999", "pilot": "#DD8452", "transfer": "#55A868"}
    for ds in datasets:
        acts = [a for a in ACT_ORDER[2:] if (ds, a, "default") in groups]
        fig, axes = plt.subplots(1, len(acts), figsize=(4 * len(acts), 3.3), sharey=True)
        for ax, act in zip(np.atleast_1d(axes), acts):
            for st in STRAT_ORDER:
                g = groups.get((ds, act, st), [])
                if not g:
                    continue
                curves = np.array([[e["val"]["loss"] for e in r["curve"]] for r in g])
                x = np.arange(1, curves.shape[1] + 1)
                mu, sd = curves.mean(0), curves.std(0)
                ax.plot(x, mu, color=colors[st], label=STRAT_NAMES[st])
                ax.fill_between(x, mu - sd, mu + sd, color=colors[st], alpha=0.15)
            for ref, ls in (("relu", ":"), ("gelu", "--")):
                g = groups.get((ds, ref, "default"), [])
                if g:
                    curves = np.array([[e["val"]["loss"] for e in r["curve"]] for r in g])
                    ax.plot(np.arange(1, curves.shape[1] + 1), curves.mean(0), color="black", ls=ls, lw=1, label=ACT_NAMES[ref])
            ax.set_title(ACT_NAMES[act])
            ax.set_xlabel("эпоха")
            ax.set_yscale("log")
        np.atleast_1d(axes)[0].set_ylabel("val loss")
        np.atleast_1d(axes)[-1].legend(fontsize=8)
        fig.suptitle(ds)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"curves_{ds}.png"), dpi=130)
        plt.close(fig)
    print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
