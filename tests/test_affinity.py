import math

import numpy as np

from src.affinity import binder_labels, ic50_nm_from_score, score_from_ic50_nm
from src.constants import BINDER_Y_THRESHOLD, BLOSUM62, IC50_CAP_NM


def test_blosum_known_entries_and_symmetry():
    # Alphabetical index: A=0, C=1, ..., W is near the end, Y last-but-one, V last? 
    # AA = ACDEFGHIKLMNPQRSTVWY so W=18, Y=19, V=17.
    assert BLOSUM62.shape == (20, 20)
    assert BLOSUM62[0, 0] == 4  # A, A
    assert BLOSUM62[1, 1] == 9  # C, C
    assert int(BLOSUM62[18, 18]) == 11  # W, W
    assert np.array_equal(BLOSUM62, BLOSUM62.T)


def test_binder_threshold_matches_ic50_500():
    at_500 = score_from_ic50_nm(np.array([500.0]))
    assert math.isclose(float(at_500[0]), BINDER_Y_THRESHOLD, rel_tol=0, abs_tol=1e-12)
    assert int(binder_labels(at_500)[0]) == 0
    assert int(binder_labels(score_from_ic50_nm(np.array([499.0])))[0]) == 1
    assert int(binder_labels(score_from_ic50_nm(np.array([501.0])))[0]) == 0
    assert int(binder_labels(np.array([0.0, 1.0]))[0]) == 0
    assert int(binder_labels(np.array([0.0, 1.0]))[1]) == 1


def test_ic50_roundtrip():
    ic50 = np.array([50.0, 500.0, 5000.0, IC50_CAP_NM])
    recovered = ic50_nm_from_score(score_from_ic50_nm(ic50))
    assert np.allclose(recovered, ic50, rtol=1e-6)
