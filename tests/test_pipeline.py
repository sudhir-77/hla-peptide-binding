import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import AA
from src.pipeline import RunConfig, run_pipeline


def _synthetic_frame(n_per: int = 80, n_alleles: int = 4) -> pd.DataFrame:
    rows = []
    row_id = 0
    for allele_i in range(n_alleles):
        allele = f"HLA-SYN{allele_i:02d}"
        mhc = AA[allele_i] * 34
        for i in range(n_per):
            binder = i % 2 == 0
            peptide = ["A"] * 9
            peptide[1] = "L" if binder else "D"
            peptide[-1] = "Y" if binder else "G"
            peptide[3] = AA[(i + allele_i) % 20]
            rows.append(
                {
                    "id": row_id,
                    "peptide": "".join(peptide),
                    "allele": allele,
                    "mhc": mhc,
                    "log_affinity": 0.85 if binder else 0.05,
                    "binder": int(binder),
                }
            )
            row_id += 1
    # One long peptide so the C-terminal clip path is exercised.
    rows[0]["peptide"] = "ACDEFGHIKLMNPQRSTVWYACDEF"
    return pd.DataFrame(rows)


def test_pipeline_writes_required_artifacts(tmp_path: Path):
    cfg = RunConfig(
        seed=0,
        n_heldout=1,
        min_allele_count=40,
        cnn_epochs=1,
        transformer_epochs=1,
        cnn_patience=1,
        transformer_patience=1,
        cnn_hidden=8,
        transformer_d_model=16,
        transformer_heads=2,
        transformer_layers=1,
        transformer_ff=32,
        batch_size=32,
        transformer_batch_size=32,
        dropout=0.0,
        demo_allele="HLA-SYN01",
    )
    results = tmp_path / "results"
    metrics = run_pipeline(_synthetic_frame(), cfg, results)
    random_csv = pd.read_csv(results / "test_predictions_random.csv")
    held_csv = pd.read_csv(results / "test_predictions_heldout.csv")
    assert list(random_csv.columns) == [
        "id",
        "peptide",
        "allele",
        "log_affinity",
        "binder",
        "model_score",
        "pssm_score",
    ]
    assert list(held_csv.columns) == ["id", "peptide", "allele", "log_affinity", "binder", "model_score"]
    summary = json.loads((results / "split_summary.json").read_text(encoding="utf-8"))
    for key in ("train_alleles", "heldout_alleles", "n_train", "n_val", "n_test_random", "n_test_heldout"):
        assert key in summary
    assert set(summary["heldout_alleles"]).isdisjoint(summary["train_alleles"])
    assert set(held_csv["allele"]).isdisjoint(summary["train_alleles"])
    assert set(random_csv["allele"]).isdisjoint(set(summary["heldout_alleles"]))
    assert random_csv["id"].is_unique
    assert held_csv["id"].is_unique
    assert set(random_csv["id"]).isdisjoint(set(held_csv["id"]))
    assert (results / "metrics.json").exists()
    assert (results / "auroc_comparison.png").exists()
    assert (results / "pred_vs_measured.png").exists()
    assert (results / "neoantigen_ranking.csv").exists()
    assert metrics["project"] == "hla-peptide-binding"
    assert metrics["seeds"] == [0]
    assert len(metrics["success_criteria"]) == 4
    assert np.isfinite(random_csv["model_score"]).all()
    assert np.isfinite(random_csv["pssm_score"]).all()
