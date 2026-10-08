import numpy as np

from src.constants import AA_TO_IDX, PAD_IDX, UNK_IDX
from src.encode import clip_peptide, sequences_to_indices


def test_clip_peptide_keeps_c_terminus():
    assert clip_peptide("ACDEFGHIK") == "ACDEFGHIK"
    clipped = clip_peptide("ABCDEFGHIJKLMNOP", max_len=15)
    assert len(clipped) == 15
    assert clipped == "ABCDEFGHIJKLMNP"
    assert clipped[-1] == "P"


def test_indices_pad_and_unknown():
    encoded = sequences_to_indices(["AC", "AX"], max_len=4)
    assert encoded.shape == (2, 4)
    assert encoded[0, 0] == AA_TO_IDX["A"]
    assert encoded[0, 1] == AA_TO_IDX["C"]
    assert encoded[0, 2] == PAD_IDX
    assert encoded[1, 1] == UNK_IDX
    assert np.all(encoded[:, 2:] == PAD_IDX)
