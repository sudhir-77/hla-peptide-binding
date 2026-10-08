import math

import numpy as np

from src.pssm import fit_pssm


def test_pssm_matches_hand_computed_log_odds():
    peptides = np.array(["AA", "AC"])
    alleles = np.array(["Z", "Z"])
    binders = np.array([1, 1])
    model = fit_pssm(peptides, alleles, binders, alpha=1.0)
    bg_a = 0.75
    p_a_pos0 = (2 + bg_a) / 3
    p_a_pos1 = (1 + bg_a) / 3
    expected = math.log(p_a_pos0 / bg_a) + math.log(p_a_pos1 / bg_a)
    assert math.isclose(model.score("AA", "Z"), expected, rel_tol=0, abs_tol=1e-9)
    assert math.isnan(model.score("AA", "OTHER"))


def test_unseen_allele_is_nan_and_missing_length_uses_anchors():
    peptides = np.array(["AA", "AC", "DD"])
    alleles = np.array(["Z", "Z", "Z"])
    binders = np.array([1, 1, 0])
    model = fit_pssm(peptides, alleles, binders, alpha=1.0)
    assert "Z" in model.seen_alleles
    longer = model.score("AAA", "Z")
    assert math.isfinite(longer)
    # No binders at all: seen allele, score 0, unseen allele is NaN.
    empty = fit_pssm(np.array(["WW"]), np.array(["Q"]), np.array([0]))
    assert empty.score("WW", "Q") == 0.0
    assert math.isnan(empty.score("WW", "NOPE"))
