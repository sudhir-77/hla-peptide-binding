import numpy as np
import pytest

from src.splits import assign_splits, choose_heldout_alleles, random_split_within_allele


def test_heldout_respects_min_count_and_is_deterministic():
    alleles = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 2)
    first = choose_heldout_alleles(alleles, n_heldout=1, min_count=5, seed=0)
    second = choose_heldout_alleles(alleles, n_heldout=1, min_count=5, seed=0)
    assert first == second
    assert first[0] in {"A", "B"}
    with pytest.raises(ValueError):
        choose_heldout_alleles(alleles, n_heldout=3, min_count=5, seed=0)


def test_within_allele_split_is_80_10_10_and_disjoint():
    alleles = np.array(["Z"] * 1000)
    train, val, test = random_split_within_allele(alleles, seed=0)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100
    assert set(train) & set(val) == set()
    assert set(train) & set(test) == set()
    assert set(train) | set(val) | set(test) == set(range(1000))
    again, _, _ = random_split_within_allele(alleles, seed=0)
    assert np.array_equal(train, again)


def test_heldout_alleles_never_enter_train_val_or_random_test():
    alleles = np.array(["A"] * 1000 + ["B"] * 800 + ["C"] * 50)
    splits = assign_splits(alleles, n_heldout=1, min_count=500, seed=0)
    groups = [splits["train"], splits["val"], splits["test_random"], splits["test_heldout"]]
    seen: set[int] = set()
    for group in groups:
        ids = set(int(i) for i in group.tolist())
        assert seen.isdisjoint(ids)
        seen |= ids
    assert seen == set(range(len(alleles)))
    held = set(splits["heldout_alleles"].tolist())
    assert len(held) == 1
    assert held <= {"A", "B"}
    for name in ("train", "val", "test_random"):
        present = set(alleles[splits[name]].tolist())
        assert present.isdisjoint(held)
    assert set(alleles[splits["test_heldout"]].tolist()) == held
    kept = [a for a in ("A", "B") if a not in held][0]
    kept_idx = splits["train"][alleles[splits["train"]] == kept]
    # The kept large allele is split near 80/10/10, so training keeps most of it.
    assert len(kept_idx) == 800 or len(kept_idx) == 640
