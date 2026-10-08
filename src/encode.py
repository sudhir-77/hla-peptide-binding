"""Peptide and HLA pseudo-sequence index encoding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.constants import AA_TO_IDX, MAX_PEP_LEN, PAD_IDX, UNK_IDX


def clip_peptide(sequence: str, max_len: int = MAX_PEP_LEN) -> str:
    """Fit a peptide into ``max_len`` while keeping the C-terminal anchor.

    MHC class I peptides in this table are 8-15 residues except for 84 rows
    (lengths 16-30). Those rare peptides keep the first ``max_len - 1``
    residues and the final residue, which is the C-terminal anchor.
    """
    sequence = sequence.strip().upper()
    if len(sequence) <= max_len:
        return sequence
    return sequence[: max_len - 1] + sequence[-1]


def sequences_to_indices(sequences: Sequence[str], max_len: int) -> np.ndarray:
    """Encode sequences as integer indices, right-padded with PAD_IDX.

    Unknown letters use UNK_IDX. They still occupy a position.
    """
    out = np.full((len(sequences), max_len), PAD_IDX, dtype=np.int64)
    for i, raw in enumerate(sequences):
        sequence = raw.strip().upper()
        for j, char in enumerate(sequence[:max_len]):
            out[i, j] = AA_TO_IDX.get(char, UNK_IDX)
    return out
