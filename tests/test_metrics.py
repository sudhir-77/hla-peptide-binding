import math

import numpy as np

from src.metrics import mean_finite, per_allele_auroc, safe_auroc, safe_spearman


def test_auroc_hand_computed():
    # Positives 0.35 and 0.8, negatives 0.1 and 0.4.
    # 0.35 beats only 0.1; 0.8 beats both. 3 of 4 pairs. AUROC = 0.75.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert math.isclose(safe_auroc(labels, scores), 0.75, abs_tol=1e-9)


def test_auroc_undefined_for_one_class():
    assert math.isnan(safe_auroc(np.array([0, 0, 0]), np.array([0.2, 0.3, 0.4])))


def test_spearman_perfect_and_reversed():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(safe_spearman(target, target), 1.0, abs_tol=1e-9)
    assert math.isclose(safe_spearman(target, target[::-1]), -1.0, abs_tol=1e-9)


def test_per_allele_mean_skips_undefined():
    alleles = np.array(["A", "A", "A", "B", "B"])
    labels = np.array([0, 1, 1, 0, 0])
    scores = np.array([0.1, 0.8, 0.9, 0.2, 0.3])
    per = per_allele_auroc(alleles, labels, scores)
    assert math.isclose(per["A"], 1.0, abs_tol=1e-9)
    assert math.isnan(per["B"])
    assert math.isclose(mean_finite([per["A"], per["B"]]), 1.0, abs_tol=1e-9)
