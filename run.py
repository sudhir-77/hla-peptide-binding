"""Reproduce the full experiment: download, train, evaluate, write results/."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from src.download import ensure_dataset, load_binding_table
from src.hardware import PeakMemory, host_summary
from src.pipeline import RunConfig, run_pipeline
from src.reporting import CRITERION_COVERAGE, criterion, update_readme, write_json, write_resume_notes
from src.splits import subsample_alleles
from src.train import limit_torch_threads

ROOT = Path(__file__).resolve().parent


def _measure_coverage(results_dir: Path) -> tuple[float, bool]:
    cov_path = results_dir / "coverage.json"
    env = os.environ.copy()
    env["HLA_SKIP_COVERAGE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not e2e",
        "--cov=src",
        f"--cov-report=json:{cov_path}",
        "-q",
    ]
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    percent = 0.0
    if cov_path.exists():
        payload = json.loads(cov_path.read_text(encoding="utf-8"))
        percent = float(payload["totals"]["percent_covered"])
    return percent, completed.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate pan-allele MHC-I binding models.")
    parser.add_argument("--quick", action="store_true", help="Small subset, a few epochs, under a few minutes.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--skip-coverage", action="store_true", help="Skip the pytest coverage measurement.")
    args = parser.parse_args(argv)

    limit_torch_threads(4)
    started = time.perf_counter()
    peak = PeakMemory()
    peak.start()
    try:
        table_path = ensure_dataset(args.data_dir)
        frame = load_binding_table(table_path)
        if len(frame) != 185_985 and not args.quick:
            raise RuntimeError(f"Expected 185985 rows after loading, found {len(frame)}.")
        cfg = RunConfig.quick(args.device) if args.quick else RunConfig.full(args.device)
        if args.quick:
            frame = subsample_alleles(frame, n_alleles=8, per_allele=180, seed=cfg.seed)
            print(f"quick subset rows={len(frame)}", flush=True)
        metrics = run_pipeline(frame, cfg, args.results_dir)
        hardware = host_summary()
        hardware["gpu"] = None if args.device == "cpu" else args.device
        metrics["hardware"] = hardware

        measure = not args.quick and not args.skip_coverage and os.environ.get("HLA_SKIP_COVERAGE") != "1"
        if measure:
            print("measuring test coverage", flush=True)
            percent, tests_ok = _measure_coverage(args.results_dir)
            metrics["success_criteria"].append(
                criterion(CRITERION_COVERAGE, percent, ">=", 75.0)
            )
            metrics["success_criteria"][-1]["passed"] = bool(
                metrics["success_criteria"][-1]["passed"] and tests_ok
            )
            metrics["metrics"]["test_coverage_percent"] = percent
            metrics["metrics"]["unit_tests_passed"] = tests_ok

        metrics["runtime_seconds"] = round(time.perf_counter() - started, 3)
        metrics["hardware"]["peak_rss_gb"] = round(peak.stop(), 4)
        write_json(args.results_dir / "metrics.json", metrics)
        if args.results_dir.resolve() == (ROOT / "results").resolve():
            write_resume_notes(ROOT / "RESUME_NOTES.md", metrics)
            update_readme(ROOT / "README.md", metrics)
    except Exception:
        peak.stop()
        raise

    print(f"runtime_seconds={metrics['runtime_seconds']}", flush=True)
    print(f"peak_rss_gb={metrics['hardware']['peak_rss_gb']}", flush=True)
    for item in metrics["success_criteria"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['name']}: {item['value']} (threshold {item['op']} {item['threshold']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
