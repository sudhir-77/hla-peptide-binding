"""Convert the TDC normalized affinity score to an IC50 binder label."""

from __future__ import annotations

import numpy as np

from src.constants import BINDER_Y_THRESHOLD, IC50_CAP_NM


def score_from_ic50_nm(ic50_nm: np.ndarray) -> np.ndarray:
    """y = 1 - log10(IC50) / log10(50000), with IC50 capped at 50,000 nM."""
    ic50 = np.clip(np.asarray(ic50_nm, dtype=np.float64), 1e-12, IC50_CAP_NM)
    return 1.0 - np.log10(ic50) / np.log10(IC50_CAP_NM)


def ic50_nm_from_score(log_affinity: np.ndarray) -> np.ndarray:
    """Invert y = 1 - log10(IC50) / log10(50000)."""
    y = np.asarray(log_affinity, dtype=np.float64)
    return np.power(IC50_CAP_NM, 1.0 - y)


def binder_labels(log_affinity: np.ndarray) -> np.ndarray:
    """1 if IC50 < 500 nM, else 0.

    Because y increases as IC50 falls, IC50 < 500 nM is y > threshold.
    Equality is IC50 == 500 nM and is not a binder.
    """
    y = np.asarray(log_affinity, dtype=np.float64)
    return (y > BINDER_Y_THRESHOLD).astype(np.int64)
