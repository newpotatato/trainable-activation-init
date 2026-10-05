"""Build results/transfer_inits.json: leave-one-dataset-out fixed inits.

For dataset D and activation A the transfer init is the candidate that the pilot
procedure picked most often on the base datasets except D (ties -> earlier candidate).
Only the four base datasets vote; a dataset added later (friedman) gets the vote of all four.
"""
import json
import os
from collections import Counter

from activations import INITS
from analyze import load_runs
from data import DATASETS
from train import TRANSFER_FILE


def main():
    base = ("california", "wine", "covertype", "fashion")
    runs = [r for r in load_runs() if r["strategy"] == "pilot" and r["dataset"] in base]
    datasets = sorted(DATASETS)
    out = {}
    for ds in datasets:
        out[ds] = {}
        for act, spec in INITS.items():
            keys = [json.dumps(c, sort_keys=True) for c in spec["candidates"]]
            votes = Counter(json.dumps(r["init"], sort_keys=True) for r in runs if r["act"] == act and r["dataset"] != ds)
            best = max(keys, key=lambda k: (votes[k], -keys.index(k)))
            out[ds][act] = json.loads(best)
    os.makedirs(os.path.dirname(TRANSFER_FILE), exist_ok=True)
    with open(TRANSFER_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
