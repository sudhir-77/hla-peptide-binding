import math
from pathlib import Path

import pytest

from src.reporting import criterion, render_results_markdown, update_readme, write_resume_notes


def _metrics() -> dict:
    return {
        "success_criteria": [
            criterion("Binder AUROC (IC50 < 500 nM) on the random test split, best model", 0.91, ">=", 0.90),
            criterion("Spearman correlation of predicted vs. measured log affinity, random test split", 0.72, ">=", 0.70),
        ],
        "metrics": {
            "n_pairs": 185985,
            "n_heldout_alleles": 10,
            "random_test": {
                "cnn": {"auroc": 0.91, "spearman": 0.72},
                "transformer": {"auroc": 0.88, "spearman": 0.70},
                "best_model": "cnn",
                "best": {"auroc": 0.91, "spearman": 0.72},
            },
            "heldout_alleles": {
                "mean_per_allele_auroc": 0.84,
                "cnn_mean_per_allele_auroc": 0.84,
                "transformer_mean_per_allele_auroc": 0.81,
            },
            "neoantigen": {"protein": "NY-ESO-1", "allele": "HLA-A02:01", "n_ninemers": 172},
        },
        "baseline": {"random_test_auroc": 0.80, "random_test_spearman": 0.55},
        "runtime_seconds": 12.5,
        "hardware": {"peak_rss_gb": 1.25},
    }


def test_criterion_pass_fail_and_nan():
    assert criterion("x", 0.90, ">=", 0.90)["passed"] is True
    assert criterion("x", 0.89, ">=", 0.90)["passed"] is False
    assert criterion("x", float("nan"), ">=", 0.90)["passed"] is False
    assert criterion("x", float("nan"), ">=", 0.90)["value"] is None
    between = criterion("x", 0.5, "between", 0.2, 0.7)
    assert between["passed"] is True
    with pytest.raises(ValueError):
        criterion("x", 1.0, "nope", 0.0)


def test_readme_and_resume_use_metric_numbers(tmp_path: Path):
    metrics = _metrics()
    readme = tmp_path / "README.md"
    readme.write_text("Intro\n\n<!-- RESULTS_START -->\nold\n<!-- RESULTS_END -->\n", encoding="utf-8")
    update_readme(readme, metrics)
    text = readme.read_text(encoding="utf-8")
    assert "0.910" in text
    assert "\nold\n" not in text
    notes = tmp_path / "RESUME_NOTES.md"
    write_resume_notes(notes, metrics)
    body = notes.read_text(encoding="utf-8")
    assert "185985" in body
    assert "0.910" in body
    assert "0.720" in body
    assert "0.800" in body
    assert "0.840" in body
    for banned in ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded", "—"):
        assert banned not in body.lower() if banned != "—" else banned not in body
    rendered = render_results_markdown(metrics)
    assert "PASS" not in rendered
    assert "yes" in rendered
    assert math.isclose(metrics["metrics"]["random_test"]["best"]["auroc"], 0.91)
