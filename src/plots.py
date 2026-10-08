"""Plots written next to the metric files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_metric_bars(labels: list[str], values: list[float], ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color="#2c6e8a")
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle="--")
    _save(fig, path)


def plot_pred_vs_measured(
    measured: np.ndarray, predicted: np.ndarray, path: Path, seed: int = 0, max_points: int = 4000
) -> None:
    rng = np.random.default_rng(seed)
    n = len(measured)
    if n > max_points:
        take = rng.choice(n, size=max_points, replace=False)
        measured = np.asarray(measured)[take]
        predicted = np.asarray(predicted)[take]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(measured, predicted, s=6, alpha=0.25, color="#2c6e8a", linewidths=0)
    ax.plot([0, 1], [0, 1], color="#444444", linewidth=1)
    ax.set_xlabel("Measured log-affinity score")
    ax.set_ylabel("Predicted score")
    ax.set_title("Random test split")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save(fig, path)


def plot_learning_curves(curves: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, values in curves.items():
        ax.plot(np.arange(1, len(values) + 1), values, marker="o", label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model selection uses this curve, not the test set")
    ax.legend()
    _save(fig, path)


def plot_per_allele(alleles: list[str], aurocs: list[float], path: Path, title: str) -> None:
    order = np.argsort(aurocs)
    names = [alleles[i] for i in order]
    vals = [aurocs[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.35 * len(names) + 1.0)))
    ax.barh(np.arange(len(names)), vals, color="#3d6b4f")
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_xlabel("AUROC")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    _save(fig, path)
