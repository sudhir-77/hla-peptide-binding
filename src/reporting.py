"""Write metrics.json, the README results table, and RESUME_NOTES.md."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

RESULTS_START = "<!-- RESULTS_START -->"
RESULTS_END = "<!-- RESULTS_END -->"

CRITERION_AUROC = "Binder AUROC (IC50 < 500 nM) on the random test split, best model"
CRITERION_SPEARMAN = "Spearman correlation of predicted vs. measured log affinity, random test split"
CRITERION_HELDOUT = "Mean per-allele AUROC on held-out alleles"
CRITERION_BASELINE = "Best model vs. PSSM baseline"
CRITERION_COVERAGE = "Test coverage of src/"


def criterion(name: str, value: float, op: str, threshold: float, threshold_high: float | None = None) -> dict:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float("nan")
    if not math.isfinite(numeric):
        passed = False
        stored: float | None = None
    else:
        stored = numeric
        if op == ">=":
            passed = stored >= threshold
        elif op == ">":
            passed = stored > threshold
        elif op == "<=":
            passed = stored <= threshold
        elif op == "<":
            passed = stored < threshold
        elif op == "between":
            high = threshold_high if threshold_high is not None else threshold
            passed = threshold <= stored <= high
        else:
            raise ValueError(f"Unknown comparison {op}")
    return {
        "name": name,
        "value": stored,
        "op": op,
        "threshold": float(threshold),
        "threshold_high": None if threshold_high is None else float(threshold_high),
        "passed": bool(passed),
    }


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n", encoding="utf-8")


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_results_markdown(metrics: dict) -> str:
    rows = [
        "| Criterion | Value | Threshold | Passed |",
        "|---|---:|---:|:---:|",
    ]
    for item in metrics["success_criteria"]:
        thr = item["threshold"]
        if item["op"] == "between" and item["threshold_high"] is not None:
            thr_text = f"{thr} to {item['threshold_high']}"
        else:
            thr_text = f"{item['op']} {thr}"
        flag = "yes" if item["passed"] else "no"
        rows.append(f"| {item['name']} | {_fmt(item['value'])} | {thr_text} | {flag} |")
    random_metrics = metrics["metrics"]["random_test"]
    held = metrics["metrics"]["heldout_alleles"]
    base = metrics["baseline"]
    detail = [
        "",
        "| Model | Random-split AUROC | Random-split Spearman | Held-out mean per-allele AUROC |",
        "|---|---:|---:|---:|",
        f"| CNN | {_fmt(random_metrics['cnn']['auroc'])} | {_fmt(random_metrics['cnn']['spearman'])} | {_fmt(held['cnn_mean_per_allele_auroc'])} |",
        f"| Transformer | {_fmt(random_metrics['transformer']['auroc'])} | {_fmt(random_metrics['transformer']['spearman'])} | {_fmt(held['transformer_mean_per_allele_auroc'])} |",
        f"| PSSM baseline | {_fmt(base['random_test_auroc'])} | {_fmt(base['random_test_spearman'])} | PSSM cannot score unseen alleles |",
        "",
        f"Best model on validation AUROC: **{random_metrics['best_model']}**.",
        f"Runtime: {_fmt(metrics.get('runtime_seconds'), 1)} seconds.",
        f"Peak RSS: {_fmt(metrics.get('hardware', {}).get('peak_rss_gb'), 3)} GB.",
    ]
    return "\n".join(rows + detail) + "\n"


def update_readme(readme_path: Path, metrics: dict) -> None:
    if not readme_path.exists():
        return
    text = readme_path.read_text(encoding="utf-8")
    block = f"{RESULTS_START}\n{render_results_markdown(metrics)}{RESULTS_END}"
    if RESULTS_START in text and RESULTS_END in text:
        start = text.index(RESULTS_START)
        end = text.index(RESULTS_END) + len(RESULTS_END)
        text = text[:start] + block + text[end:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    readme_path.write_text(text, encoding="utf-8")


def write_resume_notes(path: Path, metrics: dict) -> None:
    random_metrics = metrics["metrics"]["random_test"]
    held = metrics["metrics"]["heldout_alleles"]
    base = metrics["baseline"]
    neo = metrics["metrics"]["neoantigen"]
    n_pairs = int(metrics["metrics"]["n_pairs"])
    n_held = int(metrics["metrics"]["n_heldout_alleles"])
    auroc = _fmt(random_metrics["best"]["auroc"])
    spearman = _fmt(random_metrics["best"]["spearman"])
    pssm = _fmt(base["random_test_auroc"])
    held_auroc = _fmt(held["mean_per_allele_auroc"])
    n_mers = int(neo["n_ninemers"])
    allele = neo["allele"]
    protein = neo["protein"]
    best = random_metrics["best_model"]
    lines = [
        "# Resume notes",
        "",
        "## Draft bullets",
        "",
        (
            f"- Trained a pan-allele peptide-MHC binding model ({best}) on {n_pairs} IEDB "
            f"measurements, reaching {auroc} binder AUROC and {spearman} Spearman "
            f"vs. {pssm} for a per-allele PSSM baseline."
        ),
        (
            f"- Generalized to {n_held} alleles unseen in training with {held_auroc} "
            "mean per-allele AUROC."
        ),
        (
            f"- Built a neoantigen-ranking demo that scores every 9-mer of {protein} "
            f"({n_mers} peptides) for {allele}."
        ),
        (
            "- Compared a CNN and a small transformer on one fixed random split and one "
            "allele-held-out split, choosing the reported model by validation AUROC."
        ),
        "",
        "## Skills and tools",
        "",
        "Python, PyTorch, NumPy, pandas, scikit-learn, matplotlib, pytest, BLOSUM62, CNN, transformer.",
        "",
        "## Project summary",
        "",
        (
            "This project trains a pan-allele CNN and a small transformer to predict "
            "MHC class I binding from the peptide sequence and the MHC pseudo-sequence."
        ),
        (
            "It compares both with a per-allele PSSM on a random split and on alleles "
            f"left out of training, then ranks 9-mers from {protein}."
        ),
        "",
        "## GitHub description",
        "",
        "Pan-allele MHC class I binding prediction on IEDB affinities, with a PSSM baseline and held-out alleles.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
