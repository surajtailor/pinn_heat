"""Shared code for the 1-D heat equation PINN example.

Problem
-------
    u_t = k * u_xx,            0 < x < 1,  0 < t < 100,  k = 0.003
    u(0, t) = u(1, t) = 0
    u(x, 0) = 50 x (1 - x)

The exact solution is a Fourier sine series, which is used both to generate
training data for the standard network and to score every model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --------------------------------------------------------------------------- #
# Problem definition
# --------------------------------------------------------------------------- #
K = 0.003                      # thermal diffusivity
X_RANGE = (0.0, 1.0)           # spatial domain
T_RANGE = (0.0, 100.0)         # time domain
LB = torch.tensor([X_RANGE[0], T_RANGE[0]])   # lower bound of (x, t)
UB = torch.tensor([X_RANGE[1], T_RANGE[1]])   # upper bound of (x, t)


def initial_condition(x):
    """u(x, 0) = 50 x (1 - x)."""
    return 50.0 * x * (1.0 - x)


def exact_solution_points(x, t, k: float = K, n_terms: int = 400):
    """Exact solution evaluated element-wise at pairs (x[i], t[i]).

    Separation of variables gives, for odd n,
        u(x, t) = sum_n 400 / (n pi)^3 * sin(n pi x) * exp(-k (n pi)^2 t).
    `x` and `t` are array-likes of the same shape; the result has that shape.
    """
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = np.arange(1, n_terms + 1, 2, dtype=np.float64)           # odd modes only
    n = n.reshape((-1,) + (1,) * x.ndim)                          # broadcast over x, t
    coeff = 400.0 / (np.pi * n) ** 3
    return np.sum(coeff * np.sin(n * np.pi * x) * np.exp(-k * (n * np.pi) ** 2 * t), axis=0)


def exact_solution_grid(x, t, k: float = K, n_terms: int = 400):
    """Exact solution on the tensor-product grid x (rows) by t (columns)."""
    X, T = np.meshgrid(np.asarray(x), np.asarray(t), indexing="ij")
    return exact_solution_points(X, T, k, n_terms)


def exact_solution_torch(xt: torch.Tensor, k: float = K, n_terms: int = 400) -> torch.Tensor:
    """Differentiable version of the exact solution for an (N, 2) tensor of (x, t) rows.

    Returns an (N, 1) tensor. Computed in float64 and cast back, so gradients with
    respect to `xt` are available (needed to attack the *true* model error).
    """
    xt64 = xt.double()
    x, t = xt64[:, 0:1], xt64[:, 1:2]                                   # (N, 1)
    n = torch.arange(1, n_terms + 1, 2, dtype=torch.float64).unsqueeze(0)  # (1, M) odd modes
    coeff = 400.0 / (np.pi * n) ** 3
    u = (coeff * torch.sin(n * np.pi * x) * torch.exp(-k * (n * np.pi) ** 2 * t)).sum(dim=1, keepdim=True)
    return u.to(xt.dtype)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def grid_inputs(x, t) -> torch.Tensor:
    """Flatten a tensor-product grid into an (N, 2) float32 tensor of (x, t) pairs.

    Ordering is x-major, i.e. it matches `array.reshape(-1)` for an array of
    shape (len(x), len(t)).
    """
    X, T = torch.meshgrid(torch.as_tensor(x, dtype=torch.float32),
                          torch.as_tensor(t, dtype=torch.float32), indexing="ij")
    return torch.stack([X.reshape(-1), T.reshape(-1)], dim=1)


def to_column(a) -> torch.Tensor:
    """Array-like -> (N, 1) float32 tensor."""
    return torch.as_tensor(np.asarray(a).reshape(-1, 1), dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FCN(nn.Module):
    """Fully connected network u(x, t) with tanh activations.

    Inputs are rescaled from the problem domain to [-1, 1] inside `forward`,
    so callers always work in physical units.
    """

    def __init__(self, n_input=2, n_output=1, n_hidden=20, n_layers=4):
        super().__init__()
        self.config = dict(n_input=n_input, n_output=n_output,
                           n_hidden=n_hidden, n_layers=n_layers)
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)
        # Buffers follow the module across .to(device) and are saved in state_dict.
        self.register_buffer("lb", LB.clone())
        self.register_buffer("ub", UB.clone())

    def forward(self, xt):
        xt = 2.0 * (xt - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(xt)


def save_model(model: FCN, path: str):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path: str, map_location="cpu") -> FCN:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = FCN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
def heat_residual(model: nn.Module, xt: torch.Tensor, k: float = K) -> torch.Tensor:
    """Pointwise PDE residual  u_t - k u_xx  at the collocation points `xt` (N, 2).

    Derivatives come from automatic differentiation of the network output with
    respect to its (x, t) inputs, so each row of `xt` is treated independently.
    """
    xt = xt.clone().requires_grad_(True)
    u = model(xt)
    du = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = du[:, 0:1], du[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t - k * u_xx


# --------------------------------------------------------------------------- #
# Metrics and plotting
# --------------------------------------------------------------------------- #
def relative_l2(pred, truth) -> float:
    """||pred - truth||_2 / ||truth||_2."""
    pred, truth = np.asarray(pred).ravel(), np.asarray(truth).ravel()
    return float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))


def plot_data_layout(x, t, exact, data_xt=None, ic_xt=None, colloc_xt=None, ax=None):
    """Heat map of the exact solution with the training points overlaid."""
    ax = ax or plt.subplots(figsize=(6, 4))[1]
    im = ax.pcolormesh(x, t, exact.T, shading="auto", cmap="inferno")
    plt.colorbar(im, ax=ax, label="u (temperature)")
    if colloc_xt is not None:
        ax.scatter(colloc_xt[:, 0], colloc_xt[:, 1], s=2, c="white", alpha=0.35, label="Collocation points")
    if data_xt is not None:
        ax.scatter(data_xt[:, 0], data_xt[:, 1], s=2, c="tab:cyan", alpha=0.8, linewidths=0, label="Observed data")
    if ic_xt is not None:
        ax.scatter(ic_xt[:, 0], ic_xt[:, 1], s=14, c="tab:green", marker="s", label="Initial condition")
    ax.set_xlabel("x (position)")
    ax.set_ylabel("t (time)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    return ax


def plot_comparison(x, t, exact, preds: dict, data_t_max=None, vmax=None):
    """Rows: models. Columns: prediction, absolute error."""
    vmax = vmax or float(exact.max())
    n = len(preds)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.4 * n), squeeze=False)
    for row, (name, pred) in enumerate(preds.items()):
        err = np.maximum(np.abs(pred - exact), 1e-6)       # floor so the log scale is defined
        im0 = axes[row, 0].pcolormesh(x, t, pred.T, shading="auto", cmap="inferno", vmin=0, vmax=vmax)
        im1 = axes[row, 1].pcolormesh(x, t, err.T, shading="auto", cmap="viridis",
                                      norm=LogNorm(vmin=1e-4, vmax=vmax / 4))
        axes[row, 0].set_title(f"{name}: prediction")
        axes[row, 1].set_title(f"{name}: |error|  (rel. L2 = {relative_l2(pred, exact):.2e})")
        plt.colorbar(im0, ax=axes[row, 0], label="u")
        plt.colorbar(im1, ax=axes[row, 1], label="|u_pred - u_exact| (log scale)")
        for ax in axes[row]:
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            if data_t_max is not None:
                ax.axhline(data_t_max, color="white", ls="--", lw=1)
    fig.tight_layout()
    return fig


def plot_slices(x, t, exact, preds: dict, times, data_t_max=None):
    """Profiles u(x) at selected times for the exact solution and each model."""
    fig, axes = plt.subplots(1, len(times), figsize=(3.6 * len(times), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    styles = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
    for ax, tv in zip(axes, times):
        j = int(np.argmin(np.abs(np.asarray(t) - tv)))
        ax.plot(x, exact[:, j], "k-", lw=2, label="Exact")
        for (name, pred), c in zip(preds.items(), styles):
            ax.plot(x, pred[:, j], "--", color=c, lw=1.8, label=name)
        tag = "" if data_t_max is None else (" (training data)" if tv <= data_t_max else " (extrapolation)")
        ax.set_title(f"t = {t[j]:.0f}{tag}", fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
