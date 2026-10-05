"""One training run: MLP + activation + initialisation strategy.

Strategies for trainable activations:
  default -- the framework / paper default
  random  -- random scalar start drawn from INITS[act]["random"]
  pilot   -- ours: every candidate start is trained for PILOT_EPOCHS with the same
             weight seed, the one with the lowest val loss is then trained from
             scratch for the full budget. The pilot cost is recorded.
  transfer -- leave-one-dataset-out: the candidate the pilot picked most often on
             the other datasets (results/transfer_inits.json, made by make_transfer.py);
             answers whether a per-task pilot is needed or a better fixed default suffices.
  pilot_d -- like pilot, but the default init is added to the candidates (so the
             pilot can always fall back to the default).
  default_long -- default init trained EPOCHS + DEFAULT_LONG_EXTRA epochs: the same total
             compute as pilot5 (5 candidates x 5 epochs), for an equal-budget comparison.

Architectures: mlp3 (3 x 128, main experiments) and mlp8 (8 x 128, used to check
whether the transferred inits carry over to a deeper network).
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from activations import INITS, make_activation
from data import load, out_dim, task as dataset_task

ARCHS = {"mlp3": (128,) * 3, "mlp8": (128,) * 8}
LR = 1e-3
EPOCHS = 30
PILOT_EPOCHS = 2
DEFAULT_LONG_EXTRA = 25  # = pilot5 cost with 5 candidates
BATCH = {"california": 256, "wine": 64, "covertype": 512, "fashion": 256, "friedman": 256,
         "friedman2": 256, "friedman3": 256, "sines": 256}


class MLP(nn.Module):
    def __init__(self, d_in, d_out, act, init, hidden):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), make_activation(act, h, init)]
            d = h
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def _random_init(act, rng):
    init = {}
    for k, (lo, hi, log) in INITS[act]["random"].items():
        init[k] = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if log else float(rng.uniform(lo, hi))
    return init


def _evaluate(model, X, y, task):
    model.eval()
    with torch.no_grad():
        out = model(X)
    if task == "reg":
        mse = torch.mean((out - y) ** 2).item()
        return {"loss": mse, "rmse": math.sqrt(mse)}
    loss = nn.functional.cross_entropy(out, y).item()
    return {"loss": loss, "acc": (out.argmax(-1) == y).float().mean().item()}


def _train(d, task, dataset, act, init, seed, epochs, hidden):
    """Train from a fixed weight seed; return per-epoch val/test metrics and final params."""
    torch.manual_seed(seed)
    model = MLP(d["Xtr"].shape[1], out_dim(dataset), act, init, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss() if task == "reg" else nn.CrossEntropyLoss()
    Xtr, ytr = torch.from_numpy(d["Xtr"]), torch.from_numpy(d["ytr"])
    Xva, yva = torch.from_numpy(d["Xva"]), torch.from_numpy(d["yva"])
    Xte, yte = torch.from_numpy(d["Xte"]), torch.from_numpy(d["yte"])
    gen = torch.Generator().manual_seed(seed)
    bs = BATCH[dataset]
    curve = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        curve.append({"val": _evaluate(model, Xva, yva, task), "test": _evaluate(model, Xte, yte, task)})
    return curve, _act_params(model)


def _act_params(model):
    """Mean value of every activation parameter per layer, to see where training moved them."""
    out = []
    for m in model.net:
        names = getattr(m, "param_names", ())
        if names:
            out.append({k: float(getattr(m, k).detach().mean()) for k in names})
    return out


TRANSFER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "transfer_inits.json")


def _transfer_inits():
    with open(TRANSFER_FILE, encoding="utf-8") as f:
        return json.load(f)


def run(dataset, act, strategy, seed, arch="mlp3"):
    torch.set_num_threads(1)
    hidden = ARCHS[arch]
    task = dataset_task(dataset)
    d = load(dataset)
    t0 = time.time()
    pilot = None
    epochs = EPOCHS
    if strategy == "default":
        init = INITS[act]["default"] if act in INITS else {}
    elif strategy == "default_long":
        # equal-compute baseline for pilot5: the default init trained for the pilot's extra epochs too
        init = INITS[act]["default"] if act in INITS else {}
        epochs = EPOCHS + DEFAULT_LONG_EXTRA
    elif strategy == "random":
        init = _random_init(act, np.random.default_rng(seed))
    elif strategy == "transfer":
        # candidate picked most often by the pilot on the *other* datasets (no pilot cost here)
        init = _transfer_inits()[dataset][act]
    elif strategy in ("pilot", "pilot_d", "pilot5") or strategy.startswith("pilot_e"):
        # pilot5: longer pilot (5 epochs) with the default among the candidates, against the
        # short-horizon bias of a 2-epoch pilot; pilot_e<T>: the same with T pilot epochs (length sweep)
        if strategy.startswith("pilot_e"):
            pilot_epochs = int(strategy[len("pilot_e"):])
        else:
            pilot_epochs = 5 if strategy == "pilot5" else PILOT_EPOCHS
        cands = list(INITS[act]["candidates"])
        if strategy != "pilot" and INITS[act]["default"] not in cands:
            cands.append(INITS[act]["default"])
        scores = []
        for cand in cands:
            curve, _ = _train(d, task, dataset, act, cand, seed, pilot_epochs, hidden)
            scores.append(curve[-1]["val"]["loss"])
        init = cands[int(np.argmin(scores))]
        pilot = {"scores": scores, "extra_epochs": pilot_epochs * len(scores)}
    else:
        raise ValueError(strategy)
    curve, params = _train(d, task, dataset, act, init, seed, epochs, hidden)
    return {
        "dataset": dataset, "task": task, "act": act, "strategy": strategy, "seed": seed, "arch": arch,
        "init": init, "pilot": pilot, "curve": curve, "final_params": params,
        "seconds": time.time() - t0,
    }
