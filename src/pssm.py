"""Per-allele position-specific scoring matrix built from training binders."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.constants import AA_TO_IDX, N_AA


def _residue_index(char: str) -> int | None:
    return AA_TO_IDX.get(char)


@dataclass
class PSSM:
    """Log-odds matrices for alleles that appeared in training.

    ``matrices[(allele, length)]`` has shape (length, 20).
    ``anchors[allele]`` has shape (2, 20): row 0 is P2, row 1 is the C-terminus.
    Scores are natural-log odds versus the training background.
    Higher means more binder-like. Unseen alleles score NaN.
    """

    background: np.ndarray
    matrices: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    anchors: dict[str, np.ndarray] = field(default_factory=dict)
    seen_alleles: set[str] = field(default_factory=set)
    alpha: float = 1.0

    def score(self, peptide: str, allele: str) -> float:
        if allele not in self.seen_alleles:
            return float("nan")
        peptide = peptide.strip().upper()
        length = len(peptide)
        matrix = self.matrices.get((allele, length))
        if matrix is not None and matrix.shape[0] == length:
            total = 0.0
            for pos, char in enumerate(peptide):
                idx = _residue_index(char)
                if idx is None:
                    continue
                total += float(matrix[pos, idx])
            return total
        anchor = self.anchors.get(allele)
        if anchor is None or length < 2:
            return 0.0
        p2 = _residue_index(peptide[1])
        cterm = _residue_index(peptide[-1])
        total = 0.0
        if p2 is not None:
            total += float(anchor[0, p2])
        if cterm is not None:
            total += float(anchor[1, cterm])
        return total

    def score_many(self, peptides: np.ndarray, alleles: np.ndarray) -> np.ndarray:
        scores = np.empty(len(peptides), dtype=np.float64)
        for i, (peptide, allele) in enumerate(zip(peptides, alleles)):
            scores[i] = self.score(str(peptide), str(allele))
        return scores


def _log_odds(counts: np.ndarray, background: np.ndarray, alpha: float) -> np.ndarray:
    """counts shape (L, 20) or (20,). Returns log-odds with the same shape."""
    bg = np.maximum(background, 1e-8)
    if counts.ndim == 1:
        n = float(counts.sum())
        prob = (counts + alpha * bg) / (n + alpha)
        return np.log(prob / bg)
    n = counts.sum(axis=1, keepdims=True)
    prob = (counts + alpha * bg) / (n + alpha)
    return np.log(prob / bg)


def fit_pssm(
    peptides: np.ndarray,
    alleles: np.ndarray,
    binders: np.ndarray,
    alpha: float = 1.0,
) -> PSSM:
    """Fit a per-allele, per-length PSSM on training binders.

    Background frequencies come from every training peptide, binders and
    non-binders. Count matrices use binders only. An allele with no binders
    is still marked seen so the caller can tell "unseen" from "no motif".
    """
    peptides = np.asarray(peptides)
    alleles = np.asarray(alleles)
    binders = np.asarray(binders).astype(bool)
    bg_counts = np.zeros(N_AA, dtype=np.float64)
    for peptide in peptides:
        for char in str(peptide).strip().upper():
            idx = AA_TO_IDX.get(char)
            if idx is not None:
                bg_counts[idx] += 1.0
    total = float(bg_counts.sum())
    if total <= 0:
        background = np.full(N_AA, 1.0 / N_AA, dtype=np.float64)
    else:
        background = bg_counts / total
        background = np.maximum(background, 1e-8)

    length_counts: dict[tuple[str, int], np.ndarray] = {}
    anchor_counts: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    for peptide, allele, is_binder in zip(peptides, alleles, binders):
        allele = str(allele)
        seen.add(allele)
        if not is_binder:
            continue
        sequence = str(peptide).strip().upper()
        length = len(sequence)
        if length == 0:
            continue
        key = (allele, length)
        if key not in length_counts:
            length_counts[key] = np.zeros((length, N_AA), dtype=np.float64)
        for pos, char in enumerate(sequence):
            idx = AA_TO_IDX.get(char)
            if idx is None:
                continue
            length_counts[key][pos, idx] += 1.0
        if length >= 2:
            if allele not in anchor_counts:
                anchor_counts[allele] = np.zeros((2, N_AA), dtype=np.float64)
            p2 = AA_TO_IDX.get(sequence[1])
            cterm = AA_TO_IDX.get(sequence[-1])
            if p2 is not None:
                anchor_counts[allele][0, p2] += 1.0
            if cterm is not None:
                anchor_counts[allele][1, cterm] += 1.0

    matrices = {
        key: _log_odds(counts, background, alpha) for key, counts in length_counts.items()
    }
    anchors = {
        allele: _log_odds(counts, background, alpha)
        for allele, counts in anchor_counts.items()
    }
    return PSSM(
        background=background,
        matrices=matrices,
        anchors=anchors,
        seen_alleles=seen,
        alpha=alpha,
    )
