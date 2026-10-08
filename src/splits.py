"""Allele holdout and within-allele random splits."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def choose_heldout_alleles(
    alleles: np.ndarray,
    n_heldout: int,
    min_count: int,
    seed: int,
) -> list[str]:
    """Pick held-out alleles without using labels.

    Eligible alleles have at least ``min_count`` rows. The choice is a
    SHA-256 ranking of ``seed:allele``, so it does not depend on NumPy's
    random stream and is stable across machines.
    """
    alleles = np.asarray(alleles)
    values, counts = np.unique(alleles, return_counts=True)
    eligible = sorted(str(a) for a, c in zip(values, counts) if int(c) >= min_count)
    if len(eligible) < n_heldout:
        raise ValueError(
            f"Need {n_heldout} alleles with at least {min_count} rows, "
            f"found {len(eligible)}."
        )

    def sort_key(name: str) -> str:
        return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()

    ranked = sorted(eligible, key=sort_key)
    return sorted(ranked[:n_heldout])


def _split_one_group(indices: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 split of one allele's row positions. Truncation, not banker's round."""
    order = rng.permutation(indices.size)
    shuffled = indices[order]
    n = int(shuffled.size)
    n_test = int(n * 0.1)
    n_val = int(n * 0.1)
    n_train = n - n_val - n_test
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def random_split_within_allele(
    alleles: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each allele's rows 80/10/10.

    Stratifying by allele keeps every non-held-out allele in the training
    set, so the per-allele PSSM can score the random test split. The draw
    itself is random with a fixed seed. Alleles are visited in sorted order.
    """
    alleles = np.asarray(alleles)
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for allele in sorted(set(alleles.tolist())):
        indices = np.flatnonzero(alleles == allele)
        train, val, test = _split_one_group(indices, rng)
        if train.size:
            train_parts.append(train)
        if val.size:
            val_parts.append(val)
        if test.size:
            test_parts.append(test)

    def _cat(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts).astype(np.int64, copy=False)

    return _cat(train_parts), _cat(val_parts), _cat(test_parts)


def assign_splits(
    alleles: np.ndarray, n_heldout: int, min_count: int, seed: int
) -> dict[str, np.ndarray]:
    """Hold out alleles, then split the rest 80/10/10 within each allele.

    Returned index arrays address ``alleles`` and do not overlap.
    ``heldout_alleles`` is a 1-d array of allele names.
    """
    alleles = np.asarray(alleles)
    heldout = choose_heldout_alleles(alleles, n_heldout, min_count, seed)
    is_held = np.isin(alleles.astype(str), np.asarray(heldout, dtype=str))
    held_idx = np.flatnonzero(is_held).astype(np.int64, copy=False)
    keep_idx = np.flatnonzero(~is_held).astype(np.int64, copy=False)
    rel_train, rel_val, rel_test = random_split_within_allele(alleles[keep_idx], seed)
    return {
        "train": keep_idx[rel_train],
        "val": keep_idx[rel_val],
        "test_random": keep_idx[rel_test],
        "test_heldout": held_idx,
        "heldout_alleles": np.asarray(heldout, dtype=object),
    }


def subsample_alleles(
    df: pd.DataFrame, n_alleles: int, per_allele: int, seed: int
) -> pd.DataFrame:
    """Keep the most common alleles and a fixed number of rows from each.

    Used only by ``--quick``. The full run does not call this.
    """
    counts = df.groupby("allele", sort=False).size().sort_values(ascending=False)
    chosen = [str(a) for a in counts.head(n_alleles).index.tolist()]
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for allele in chosen:
        block = df.loc[df["allele"] == allele]
        take = min(per_allele, len(block))
        picked = rng.choice(block.index.to_numpy(), size=take, replace=False)
        parts.append(df.loc[picked])
    out = pd.concat(parts, axis=0)
    return out.reset_index(drop=True)
