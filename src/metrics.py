"""AUROC and Spearman helpers, including per-allele averages."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUROC, or NaN when a split has only one class or no rows."""
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Spearman correlation, or NaN when it is undefined."""
    target = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(target) & np.isfinite(scores)
    target = target[mask]
    scores = scores[mask]
    if target.size < 2:
        return float("nan")
    coef = spearmanr(target, scores).statistic
    if coef is None or (isinstance(coef, float) and math.isnan(coef)):
        return float("nan")
    return float(coef)


def per_allele_auroc(
    alleles: np.ndarray, y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    """AUROC for each allele. Alleles with one class map to NaN."""
    alleles = np.asarray(alleles)
    out: dict[str, float] = {}
    for allele in sorted(set(alleles.tolist())):
        mask = alleles == allele
        out[str(allele)] = safe_auroc(y_true[mask], y_score[mask])
    return out


def mean_finite(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())
