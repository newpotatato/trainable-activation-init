"""Run the full grid in parallel and append results to results/runs.jsonl.

Already finished runs are skipped, so the script can be stopped and restarted.

    python run_all.py                 # full grid
    python run_all.py --seeds 1 --datasets wine         # quick smoke test
    python run_all.py --arch mlp8 --acts prelu swish siglin --fixed relu sigmoid         --strategies default transfer --out results/runs_mlp8.jsonl
"""
import argparse
import json
import os
from multiprocessing import Pool

from activations import FIXED, TRAINABLE
from data import DATASETS, prepare
from train import run

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "runs.jsonl")


def _job(args):
    try:
        return run(*args)
    except Exception as e:  # keep the pool alive, report the failure
        return {"error": repr(e), "job": list(args)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument("--out", default=RESULTS)
    p.add_argument("--strategies", nargs="+", default=["default", "random", "pilot"])
    p.add_argument("--acts", nargs="+", default=list(TRAINABLE), help="trainable activations to run")
    p.add_argument("--fixed", nargs="+", default=["relu", "gelu"], help="fixed reference activations (only with 'default')")
    p.add_argument("--arch", default="mlp3")
    a = p.parse_args()

    for name in a.datasets:
        prepare(name)  # download/cache once before forking workers

    jobs = []
    for ds in a.datasets:
        for seed in range(a.seeds):
            if "default" in a.strategies:
                for act in a.fixed:
                    jobs.append((ds, act, "default", seed, a.arch))
            for act in a.acts:
                for strategy in a.strategies:
                    jobs.append((ds, act, strategy, seed, a.arch))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r:
                    done.add((r["dataset"], r["act"], r["strategy"], r["seed"], r.get("arch", "mlp3")))
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run on {a.workers} workers", flush=True)

    with Pool(a.workers) as pool, open(a.out, "a", encoding="utf-8") as f:
        for i, r in enumerate(pool.imap_unordered(_job, todo), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            tag = r.get("error") or f'{r["dataset"]:<10} {r["act"]:<6} {r["strategy"]:<7} s{r["seed"]} {r["seconds"]:.0f}s'
            print(f"[{i}/{len(todo)}] {tag}", flush=True)


if __name__ == "__main__":
    main()
