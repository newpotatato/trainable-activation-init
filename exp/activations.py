"""Trainable activation functions with per-neuron parameters.

Every trainable activation takes `width` (number of neurons in the layer) and an
`init` dict of scalar starting values; each neuron gets its own copy of every
parameter, initialised to the same scalar.
"""
import torch
import torch.nn as nn


class PReLU(nn.Module):
    """He et al., 2015: f(x) = x if x >= 0 else a*x."""
    param_names = ("a",)

    def __init__(self, width, init):
        super().__init__()
        self.a = nn.Parameter(torch.full((width,), float(init["a"])))

    def forward(self, x):
        return torch.where(x >= 0, x, self.a * x)


class SwishBeta(nn.Module):
    """Ramachandran et al., 2017: f(x) = x * sigmoid(beta * x)."""
    param_names = ("beta",)

    def __init__(self, width, init):
        super().__init__()
        self.beta = nn.Parameter(torch.full((width,), float(init["beta"])))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class AconC(nn.Module):
    """Ma et al., 2021: f(x) = (p1-p2)*x*sigmoid(beta*(p1-p2)*x) + p2*x.

    Default init follows the official code: p1, p2 ~ N(0, 1), beta = 1.
    """
    param_names = ("p1", "p2", "beta")

    def __init__(self, width, init):
        super().__init__()
        self.p1 = nn.Parameter(_init_tensor(width, init["p1"]))
        self.p2 = nn.Parameter(_init_tensor(width, init["p2"]))
        self.beta = nn.Parameter(_init_tensor(width, init["beta"]))

    def forward(self, x):
        d = (self.p1 - self.p2) * x
        return d * torch.sigmoid(self.beta * d) + self.p2 * x


class SigLinT(nn.Module):
    """Trainable SigLin (ours): f(x) = n*sigmoid(B*x) + L*x.

    f'(x) lies in [L, L + n*B/4], so the gradient never vanishes when L > 0.
    """
    param_names = ("B", "n", "L")

    def __init__(self, width, init):
        super().__init__()
        self.B = nn.Parameter(torch.full((width,), float(init["B"])))
        self.n = nn.Parameter(torch.full((width,), float(init["n"])))
        self.L = nn.Parameter(torch.full((width,), float(init["L"])))

    def forward(self, x):
        return self.n * torch.sigmoid(self.B * x) + self.L * x


def _init_tensor(width, value):
    # "randn" reproduces the official ACON-C initialisation
    if value == "randn":
        return torch.randn(width)
    return torch.full((width,), float(value))


class Fixed(nn.Module):
    """Wrapper for non-trainable reference activations (ReLU, GELU)."""
    param_names = ()

    def __init__(self, width, init, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


TRAINABLE = {
    "prelu": PReLU,
    "swish": SwishBeta,
    "aconc": AconC,
    "siglin": SigLinT,
}

FIXED = {
    "relu": torch.relu,
    "gelu": nn.functional.gelu,
    "sigmoid": torch.sigmoid,
}

# Initialisation strategies for every trainable activation.
#   default    -- the value used in the original paper / framework
#   candidates -- grid evaluated by the pilot procedure
#   random     -- ranges for a random scalar start (log=True -> log-uniform)
INITS = {
    "prelu": {
        "default": {"a": 0.25},
        "candidates": [{"a": 0.01}, {"a": 0.1}, {"a": 0.25}, {"a": 0.5}],
        "random": {"a": (0.0, 0.5, False)},
    },
    "swish": {
        "default": {"beta": 1.0},
        "candidates": [{"beta": 0.5}, {"beta": 1.0}, {"beta": 2.0}, {"beta": 4.0}],
        "random": {"beta": (0.25, 4.0, True)},
    },
    "aconc": {
        "default": {"p1": "randn", "p2": "randn", "beta": 1.0},
        "candidates": [
            {"p1": 1.0, "p2": 0.0, "beta": 1.0},
            {"p1": 1.0, "p2": 0.25, "beta": 1.0},
            {"p1": 1.0, "p2": 0.0, "beta": 4.0},
            {"p1": 1.0, "p2": 0.25, "beta": 4.0},
        ],
        "random": {"p1": (0.5, 1.5, False), "p2": (0.0, 0.5, False), "beta": (0.25, 4.0, True)},
    },
    "siglin": {
        # fixed a priori, not tuned on results: unit slope at zero like standard
        # activations, f'(0) = n*B/4 + L = 1 with derivative floor L = 0.1, n = 1
        "default": {"B": 3.6, "n": 1.0, "L": 0.1},
        "candidates": [
            {"B": 1.0, "n": 1.0, "L": 0.05},
            {"B": 1.0, "n": 1.0, "L": 0.2},
            {"B": 4.0, "n": 1.0, "L": 0.05},
            {"B": 4.0, "n": 1.0, "L": 0.2},
        ],
        "random": {"B": (0.25, 4.0, True), "n": (0.5, 2.0, True), "L": (0.0, 0.3, False)},
    },
}


def make_activation(name, width, init):
    if name in TRAINABLE:
        return TRAINABLE[name](width, init)
    return Fixed(width, init, FIXED[name])
