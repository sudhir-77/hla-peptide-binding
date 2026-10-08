"""Slide a window over a protein and score every 9-mer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.encode import clip_peptide, sequences_to_indices
from src.train import predict


def resolve_demo_allele(frame: pd.DataFrame, preferred: str) -> tuple[str, str]:
    """Return (allele, pseudo-sequence). Fall back to the most common allele."""
    if (frame["allele"] == preferred).any():
        allele = preferred
    else:
        allele = str(frame["allele"].value_counts().index[0])
    pseudo = str(frame.loc[frame["allele"] == allele, "mhc"].iloc[0])
    return allele, pseudo


def sliding_ninemers(sequence: str) -> list[tuple[int, str]]:
    """1-based start position and peptide for every contiguous 9-mer."""
    text = "".join(sequence.split()).upper()
    return [(i + 1, text[i : i + 9]) for i in range(len(text) - 8)]


def rank_ninemers(
    model: torch.nn.Module,
    sequence: str,
    mhc_sequence: str,
    allele: str,
    device: torch.device,
    max_pep_len: int,
    max_mhc_len: int,
) -> pd.DataFrame:
    """Score every 9-mer. Rank 1 is the strongest predicted binder."""
    pairs = sliding_ninemers(sequence)
    if not pairs:
        return pd.DataFrame(columns=["start", "peptide", "allele", "model_score", "rank"])
    peptides = [clip_peptide(pep, max_pep_len) for _, pep in pairs]
    pep_idx = sequences_to_indices(peptides, max_pep_len)
    mhc_idx = sequences_to_indices([mhc_sequence] * len(peptides), max_mhc_len)
    scores = predict(model, pep_idx, mhc_idx, device, batch_size=256)
    frame = pd.DataFrame(
        {
            "start": [start for start, _ in pairs],
            "peptide": [pep for _, pep in pairs],
            "allele": allele,
            "model_score": scores,
        }
    )
    frame = frame.sort_values(["model_score", "start"], ascending=[False, True], kind="mergesort")
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)
