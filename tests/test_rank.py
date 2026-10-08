import pandas as pd
import torch

from src.models import PanAlleleCNN
from src.rank import rank_ninemers, resolve_demo_allele, sliding_ninemers
from src.sample_protein import KNOWN_A0201_EPITOPE, NY_ESO1_SEQUENCE


def test_ninemers_include_known_epitope():
    pairs = sliding_ninemers(NY_ESO1_SEQUENCE)
    assert len(pairs) == len(NY_ESO1_SEQUENCE) - 8
    assert KNOWN_A0201_EPITOPE in {pep for _, pep in pairs}
    assert sliding_ninemers("AAAA") == []
    assert sliding_ninemers("  ACDEFGHIK  ")[0] == (1, "ACDEFGHIK")


def test_resolve_demo_allele_prefers_requested_then_falls_back():
    frame = pd.DataFrame(
        {
            "allele": ["HLA-A02:01", "HLA-B07:02", "HLA-B07:02"],
            "mhc": ["A" * 34, "C" * 34, "C" * 34],
        }
    )
    assert resolve_demo_allele(frame, "HLA-A02:01") == ("HLA-A02:01", "A" * 34)
    allele, pseudo = resolve_demo_allele(frame, "HLA-MISSING")
    assert allele == "HLA-B07:02"
    assert pseudo == "C" * 34


def test_rank_ninemers_orders_scores():
    model = PanAlleleCNN(hidden=8, dropout=0.0)
    ranked = rank_ninemers(model, "ACDEFGHIKLMNPQRST", "A" * 34, "HLA-A02:01", torch.device("cpu"), 15, 34)
    assert len(ranked) == 9
    assert list(ranked["rank"]) == list(range(1, 10))
    assert ranked["model_score"].is_monotonic_decreasing
    empty = rank_ninemers(model, "SHORT", "A" * 34, "HLA-A02:01", torch.device("cpu"), 15, 34)
    assert empty.empty
