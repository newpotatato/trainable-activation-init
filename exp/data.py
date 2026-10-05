"""Dataset loading and preprocessing.

Every dataset is split once (seed 0) into train/val/test = 70/15/15 (or the
official test set for Fashion-MNIST), features are standardised on train,
regression targets too. Prepared arrays are cached in data/<name>.npz so worker
processes never download anything.
"""
import gzip
import os
import urllib.request

import numpy as np

# pandas/sklearn are imported lazily inside the preparation functions so that
# worker processes, which only read cached .npz files, stay light on memory.

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# task: "reg" (MSE, report RMSE in target-std units) or "cls" (cross-entropy, report accuracy)
DATASETS = {
    "california": {"task": "reg"},
    "wine": {"task": "reg"},
    "covertype": {"task": "cls", "classes": 7},
    "fashion": {"task": "cls", "classes": 10},
    # added later as an out-of-distribution task for the transferred inits (smooth synthetic target)
    "friedman": {"task": "reg"},
    # more smooth synthetic regressions, to check whether the friedman result is a pattern
    "friedman2": {"task": "reg"},
    "friedman3": {"task": "reg"},
    "sines": {"task": "reg"},
}

COVERTYPE_SUBSAMPLE = 200_000
FASHION_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"


def task(name):
    return DATASETS[name]["task"]


def out_dim(name):
    return DATASETS[name].get("classes", 1)


def _raw_california():
    from sklearn.datasets import fetch_california_housing

    d = fetch_california_housing(data_home=DATA_DIR)
    return d.data.astype(np.float64), d.target.astype(np.float64)


def _raw_wine():
    """UCI Wine Quality, red + white (6497 wines), target = quality score; colour is a feature."""
    import pandas as pd

    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    parts = []
    for colour, flag in (("red", 1.0), ("white", 0.0)):
        path = _download(base + f"winequality-{colour}.csv", f"winequality-{colour}.csv")
        df = pd.read_csv(path, sep=";")
        df["is_red"] = flag
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    y = df.pop("quality").astype(np.float64).values
    return df.values.astype(np.float64), y


def _raw_covertype():
    """UCI Covertype (7 forest cover classes), stratified subsample of 200k rows."""
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split

    d = fetch_covtype(data_home=DATA_DIR)
    X, y = d.data.astype(np.float64), d.target.astype(np.int64) - 1
    X, _, y, _ = train_test_split(X, y, train_size=COVERTYPE_SUBSAMPLE, random_state=0, stratify=y)
    return X, y


def _download(url, fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def _fashion_part(kind):
    with gzip.open(_download(FASHION_URL + f"{kind}-images-idx3-ubyte.gz", f"fashion-{kind}-images.gz")) as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64) / 255.0
    with gzip.open(_download(FASHION_URL + f"{kind}-labels-idx1-ubyte.gz", f"fashion-{kind}-labels.gz")) as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)
    return X, y


def _split_fashion():
    """Official 60k/10k split; 10k of the official train go to validation."""
    from sklearn.model_selection import train_test_split

    X, y = _fashion_part("train")
    Xte, yte = _fashion_part("t10k")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=10_000, random_state=0, stratify=y)
    return Xtr, Xva, Xte, ytr, yva, yte


def _raw_friedman():
    """Friedman #1: y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + noise; x6..x10 are irrelevant."""
    from sklearn.datasets import make_friedman1

    X, y = make_friedman1(n_samples=20_000, n_features=10, noise=1.0, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


def _raw_friedman2():
    """Friedman #2 (noise sd 125 as in Breiman, 1996)."""
    from sklearn.datasets import make_friedman2

    X, y = make_friedman2(n_samples=20_000, noise=125.0, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


def _raw_friedman3():
    """Friedman #3 (noise sd 0.1 as in Breiman, 1996)."""
    from sklearn.datasets import make_friedman3

    X, y = make_friedman3(n_samples=20_000, noise=0.1, random_state=0)
    return X.astype(np.float64), y.astype(np.float64)


def _raw_sines():
    """Smooth oscillating target: y = sin(2 pi x1) cos(pi x2) + sin(3 x3 x4) + 0.5 x5^2 + noise; x ~ U(-1, 1)^8."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(20_000, 8))
    y = (np.sin(2 * np.pi * X[:, 0]) * np.cos(np.pi * X[:, 1]) + np.sin(3 * X[:, 2] * X[:, 3])
         + 0.5 * X[:, 4] ** 2 + rng.normal(0, 0.1, 20_000))
    return X, y


RAW = {"california": _raw_california, "wine": _raw_wine, "covertype": _raw_covertype, "friedman": _raw_friedman,
       "friedman2": _raw_friedman2, "friedman3": _raw_friedman3, "sines": _raw_sines}


def prepare(name):
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if os.path.exists(path):
        return path
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if name == "fashion":
        Xtr, Xva, Xte, ytr, yva, yte = _split_fashion()
    else:
        X, y = RAW[name]()
        strat = y if task(name) == "cls" else None
        Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, random_state=0, stratify=strat)
        strat = ytmp if task(name) == "cls" else None
        Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, random_state=0, stratify=strat)
    sx = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = (sx.transform(a) for a in (Xtr, Xva, Xte))
    if task(name) == "reg":
        mu, sd = ytr.mean(), ytr.std()
        ytr, yva, yte = ((a - mu) / sd for a in (ytr, yva, yte))
        ytype = np.float32
    else:
        ytype = np.int64
    np.savez_compressed(
        path,
        Xtr=Xtr.astype(np.float32), ytr=ytr.astype(ytype),
        Xva=Xva.astype(np.float32), yva=yva.astype(ytype),
        Xte=Xte.astype(np.float32), yte=yte.astype(ytype),
    )
    return path


def load(name):
    d = np.load(prepare(name))
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    for name in DATASETS:
        d = load(name)
        print(name, DATASETS[name], {k: v.shape for k, v in d.items()})
