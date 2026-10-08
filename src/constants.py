"""Shared constants: alphabet, BLOSUM62, dataset identity, and thresholds."""

from __future__ import annotations

import math

import numpy as np

# Direct download of the file PyTDC fetches for MHC1_IEDB-IMGT_Nielsen.
# name2id in TDC metadata maps this dataset to Harvard Dataverse file 4167073.
DATA_URL = "https://dataverse.harvard.edu/api/access/datafile/4167073"
DATA_FILENAME = "mhc1_iedb-imgt_nielsen.tab"
EXPECTED_ROWS = 185_985
EXPECTED_SHA256 = "5e57c1b4c144441d36e112751db0b0c8ff7b0fb3da40ae39dc0447e7321685ef"

# TDC column names in the tab file. X1/X2 are the two sequences, ID2 is the allele.
COL_ID = "ID1"
COL_PEPTIDE = "X1"
COL_ALLELE = "ID2"
COL_MHC = "X2"
COL_TARGET = "Y"

# Standard amino acids, alphabetical. Model indices follow this order.
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA)}
UNK_IDX = 20
PAD_IDX = 21
N_AA = 20

# NetMHCpan caps measured IC50 at 50,000 nM and trains on
# y = 1 - log10(IC50) / log10(50000). The TDC column Y is already this value.
IC50_CAP_NM = 50_000.0
IC50_BINDER_NM = 500.0
BINDER_Y_THRESHOLD = 1.0 - math.log10(IC50_BINDER_NM) / math.log10(IC50_CAP_NM)

MAX_PEP_LEN = 15
MHC_LEN = 34
# BLOSUM integers are large next to one-hot 0/1. Divide so both are O(1).
BLOSUM_SCALE = 5.0

# NCBI BLOSUM62 row order (not alphabetical).
_BLOSUM_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM_ROWS: dict[str, list[int]] = {
    "A": [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    "R": [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
    "N": [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
    "D": [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
    "C": [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
    "Q": [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
    "E": [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
    "G": [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
    "H": [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
    "I": [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
    "L": [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
    "K": [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
    "M": [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
    "F": [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
    "P": [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
    "S": [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
    "T": [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
    "W": [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
    "Y": [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
    "V": [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],
}


def blosum62_matrix() -> np.ndarray:
    """Return BLOSUM62 reindexed to the alphabetical amino-acid order."""
    src = {aa: i for i, aa in enumerate(_BLOSUM_ORDER)}
    matrix = np.zeros((N_AA, N_AA), dtype=np.float32)
    for i, aa in enumerate(AA):
        row = _BLOSUM_ROWS[aa]
        for j, bb in enumerate(AA):
            matrix[i, j] = float(row[src[bb]])
    return matrix


BLOSUM62 = blosum62_matrix()
