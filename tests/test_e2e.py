import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.e2e
def test_quick_run_writes_schema(tmp_path: Path):
    results = tmp_path / "results"
    env = os.environ.copy()
    env["HLA_SKIP_COVERAGE"] = "1"
    env["OMP_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--quick",
            "--results-dir",
            str(results),
            "--data-dir",
            str(ROOT / "data"),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((results / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["n_train"] > 0
    assert summary["n_test_random"] > 0
    assert summary["n_test_heldout"] > 0
    assert set(summary["heldout_alleles"]).isdisjoint(summary["train_alleles"])
    random_csv = pd.read_csv(results / "test_predictions_random.csv")
    held_csv = pd.read_csv(results / "test_predictions_heldout.csv")
    assert "id" in random_csv.columns
    assert set(held_csv["allele"]).isdisjoint(set(summary["train_alleles"]))
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["project"] == "hla-peptide-binding"
    assert "runtime_seconds" in metrics
    assert metrics["hardware"]["peak_rss_gb"] > 0
