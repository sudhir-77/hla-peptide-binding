"""End-to-end training and evaluation. The test split is scored once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.constants import MAX_PEP_LEN, MHC_LEN
from src.encode import clip_peptide, sequences_to_indices
from src.metrics import mean_finite, per_allele_auroc, safe_auroc, safe_spearman
from src.models import PanAlleleCNN, SmallTransformer
from src.plots import plot_learning_curves, plot_metric_bars, plot_per_allele, plot_pred_vs_measured
from src.pssm import fit_pssm
from src.rank import rank_ninemers, resolve_demo_allele
from src.reporting import (
    CRITERION_AUROC,
    CRITERION_BASELINE,
    CRITERION_HELDOUT,
    CRITERION_SPEARMAN,
    criterion,
    write_json,
)
from src.sample_protein import (
    DEMO_ALLELE,
    KNOWN_A0201_EPITOPE,
    NY_ESO1_NAME,
    NY_ESO1_SEQUENCE,
    NY_ESO1_UNIPROT,
)
from src.splits import assign_splits
from src.train import predict, set_seed, train_regressor


@dataclass
class RunConfig:
    seed: int = 0
    device: str = "cpu"
    n_heldout: int = 10
    min_allele_count: int = 500
    max_pep_len: int = MAX_PEP_LEN
    max_mhc_len: int = MHC_LEN
    batch_size: int = 512
    cnn_epochs: int = 12
    transformer_epochs: int = 6
    cnn_patience: int = 3
    transformer_patience: int = 2
    cnn_hidden: int = 128
    cnn_lr: float = 1e-3
    transformer_d_model: int = 48
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_ff: int = 96
    transformer_lr: float = 8e-4
    transformer_batch_size: int = 256
    dropout: float = 0.1
    demo_allele: str = DEMO_ALLELE

    @classmethod
    def full(cls, device: str = "cpu") -> RunConfig:
        return cls(device=device)

    @classmethod
    def quick(cls, device: str = "cpu") -> RunConfig:
        return cls(
            device=device,
            n_heldout=2,
            min_allele_count=80,
            batch_size=128,
            cnn_epochs=2,
            transformer_epochs=2,
            cnn_patience=2,
            transformer_patience=2,
            cnn_hidden=16,
            transformer_d_model=16,
            transformer_heads=2,
            transformer_layers=1,
            transformer_ff=32,
            transformer_batch_size=128,
            dropout=0.0,
        )


def _device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def _encode(frame: pd.DataFrame, max_pep_len: int, max_mhc_len: int) -> tuple[np.ndarray, np.ndarray]:
    peptides = [clip_peptide(str(seq), max_pep_len) for seq in frame["peptide"].tolist()]
    pep = sequences_to_indices(peptides, max_pep_len)
    mhc = sequences_to_indices([str(seq) for seq in frame["mhc"].tolist()], max_mhc_len)
    return pep, mhc


def _allele_table(
    alleles: np.ndarray,
    binders: np.ndarray,
    model_scores: np.ndarray,
    pssm_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for allele, auroc in per_allele_auroc(alleles, binders, model_scores).items():
        mask = alleles == allele
        row: dict[str, object] = {
            "allele": allele,
            "n": int(mask.sum()),
            "n_binders": int(binders[mask].sum()),
            "model_auroc": auroc,
        }
        if pssm_scores is not None:
            row["pssm_auroc"] = safe_auroc(binders[mask], pssm_scores[mask])
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_frame(
    frame: pd.DataFrame,
    indices: np.ndarray,
    model_scores: np.ndarray,
    pssm_scores: np.ndarray | None,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "id": frame["id"].to_numpy()[indices],
            "peptide": frame["peptide"].to_numpy()[indices],
            "allele": frame["allele"].to_numpy()[indices],
            "log_affinity": frame["log_affinity"].to_numpy()[indices],
            "binder": frame["binder"].to_numpy()[indices].astype(int),
            "model_score": model_scores,
        }
    )
    if pssm_scores is not None:
        out["pssm_score"] = pssm_scores
    return out.sort_values("id", kind="mergesort")


def run_pipeline(frame: pd.DataFrame, cfg: RunConfig, results_dir: Path) -> dict:
    """Train, score the held-out splits once, and write evaluation files.

    ``frame`` must already contain id, peptide, allele, mhc, log_affinity, binder.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    device = _device(cfg.device)
    alleles = frame["allele"].to_numpy()
    splits = assign_splits(alleles, cfg.n_heldout, cfg.min_allele_count, cfg.seed)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test_random"]
    held_idx = splits["test_heldout"]
    heldout_alleles = [str(a) for a in splits["heldout_alleles"].tolist()]
    _assert_split_integrity(alleles, train_idx, val_idx, test_idx, held_idx, heldout_alleles)

    print(
        f"splits train={len(train_idx)} val={len(val_idx)} "
        f"test_random={len(test_idx)} test_heldout={len(held_idx)} "
        f"heldout_alleles={', '.join(heldout_alleles)}",
        flush=True,
    )

    pep_idx, mhc_idx = _encode(frame, cfg.max_pep_len, cfg.max_mhc_len)
    y = frame["log_affinity"].to_numpy(dtype=np.float64)
    binders = frame["binder"].to_numpy(dtype=np.int64)
    peptides = frame["peptide"].to_numpy()

    pssm = fit_pssm(peptides[train_idx], alleles[train_idx], binders[train_idx])
    print("training cnn", flush=True)
    set_seed(cfg.seed)
    cnn = PanAlleleCNN(hidden=cfg.cnn_hidden, dropout=cfg.dropout)
    cnn_hist = train_regressor(
        cnn,
        pep_idx[train_idx],
        mhc_idx[train_idx],
        y[train_idx],
        binders[train_idx],
        pep_idx[val_idx],
        mhc_idx[val_idx],
        y[val_idx],
        binders[val_idx],
        epochs=cfg.cnn_epochs,
        patience=cfg.cnn_patience,
        batch_size=cfg.batch_size,
        lr=cfg.cnn_lr,
        device=device,
        seed=cfg.seed,
    )
    print("training transformer", flush=True)
    set_seed(cfg.seed)
    transformer = SmallTransformer(
        d_model=cfg.transformer_d_model,
        nhead=cfg.transformer_heads,
        nlayers=cfg.transformer_layers,
        dim_feedforward=cfg.transformer_ff,
        dropout=cfg.dropout,
        max_pep_len=cfg.max_pep_len,
        max_mhc_len=cfg.max_mhc_len,
    )
    transformer_hist = train_regressor(
        transformer,
        pep_idx[train_idx],
        mhc_idx[train_idx],
        y[train_idx],
        binders[train_idx],
        pep_idx[val_idx],
        mhc_idx[val_idx],
        y[val_idx],
        binders[val_idx],
        epochs=cfg.transformer_epochs,
        patience=cfg.transformer_patience,
        batch_size=cfg.transformer_batch_size,
        lr=cfg.transformer_lr,
        device=device,
        seed=cfg.seed,
    )

    val_choice = {
        "cnn": (cnn_hist.best_val_auroc, cnn_hist.val_spearman[cnn_hist.best_epoch - 1]),
        "transformer": (
            transformer_hist.best_val_auroc,
            transformer_hist.val_spearman[transformer_hist.best_epoch - 1],
        ),
    }
    best_name = max(
        val_choice,
        key=lambda name: (
            val_choice[name][0] if np.isfinite(val_choice[name][0]) else -1.0,
            val_choice[name][1] if np.isfinite(val_choice[name][1]) else -1.0,
            name == "cnn",
        ),
    )
    models = {"cnn": cnn, "transformer": transformer}
    print(f"selected {best_name} on validation AUROC", flush=True)

    test_scores = {
        name: predict(model, pep_idx[test_idx], mhc_idx[test_idx], device) for name, model in models.items()
    }
    held_scores = {
        name: predict(model, pep_idx[held_idx], mhc_idx[held_idx], device) for name, model in models.items()
    }
    pssm_test = pssm.score_many(peptides[test_idx], alleles[test_idx])
    # Held-out alleles were removed before training, so the PSSM has no matrix for them.
    pssm_held = pssm.score_many(peptides[held_idx], alleles[held_idx])
    if np.isfinite(pssm_held).any():
        raise RuntimeError("PSSM produced a finite score for a held-out allele.")

    best_test = test_scores[best_name]
    best_held = held_scores[best_name]
    shared = np.isfinite(pssm_test)
    model_on_shared = safe_auroc(binders[test_idx][shared], best_test[shared])
    pssm_auroc = safe_auroc(binders[test_idx][shared], pssm_test[shared])
    pssm_spearman = safe_spearman(y[test_idx][shared], pssm_test[shared])
    gap = model_on_shared - pssm_auroc

    random_model_metrics = {}
    held_means = {}
    for name in ("cnn", "transformer"):
        random_model_metrics[name] = {
            "auroc": safe_auroc(binders[test_idx], test_scores[name]),
            "spearman": safe_spearman(y[test_idx], test_scores[name]),
            "mean_per_allele_auroc": mean_finite(
                list(per_allele_auroc(alleles[test_idx], binders[test_idx], test_scores[name]).values())
            ),
        }
        held_per = per_allele_auroc(alleles[held_idx], binders[held_idx], held_scores[name])
        held_means[name] = mean_finite(list(held_per.values()))
    best_held_per = per_allele_auroc(alleles[held_idx], binders[held_idx], best_held)
    best_held_mean = mean_finite(list(best_held_per.values()))

    _prediction_frame(frame, test_idx, best_test, pssm_test).to_csv(
        results_dir / "test_predictions_random.csv", index=False
    )
    _prediction_frame(frame, held_idx, best_held, None).to_csv(
        results_dir / "test_predictions_heldout.csv", index=False
    )
    random_table = _allele_table(alleles[test_idx], binders[test_idx], best_test, pssm_test)
    held_table = _allele_table(alleles[held_idx], binders[held_idx], best_held, None)
    random_table.to_csv(results_dir / "per_allele_auroc_random.csv", index=False)
    held_table.to_csv(results_dir / "per_allele_auroc_heldout.csv", index=False)

    train_alleles = sorted(set(alleles[train_idx].astype(str).tolist()))
    summary = {
        "train_alleles": train_alleles,
        "heldout_alleles": heldout_alleles,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test_random": int(len(test_idx)),
        "n_test_heldout": int(len(held_idx)),
    }
    write_json(results_dir / "split_summary.json", summary)

    history_rows = []
    for name, hist in (("cnn", cnn_hist), ("transformer", transformer_hist)):
        for epoch, (loss, vloss, vauc, vsp) in enumerate(
            zip(hist.train_loss, hist.val_loss, hist.val_auroc, hist.val_spearman), start=1
        ):
            history_rows.append(
                {
                    "model": name,
                    "epoch": epoch,
                    "train_loss": loss,
                    "val_loss": vloss,
                    "val_auroc": vauc,
                    "val_spearman": vsp,
                }
            )
    pd.DataFrame(history_rows).to_csv(results_dir / "training_history.csv", index=False)

    demo_allele, demo_mhc = resolve_demo_allele(frame, cfg.demo_allele)
    ranked = rank_ninemers(
        models[best_name],
        NY_ESO1_SEQUENCE,
        demo_mhc,
        demo_allele,
        device,
        cfg.max_pep_len,
        cfg.max_mhc_len,
    )
    ranked.to_csv(results_dir / "neoantigen_ranking.csv", index=False)
    epitope_rows = ranked.loc[ranked["peptide"] == KNOWN_A0201_EPITOPE]
    if epitope_rows.empty:
        epitope_rank = None
        epitope_score = None
    else:
        epitope_rank = int(epitope_rows.iloc[0]["rank"])
        epitope_score = float(epitope_rows.iloc[0]["model_score"])

    plot_metric_bars(
        ["PSSM", "CNN", "Transformer", "Held-out mean"],
        [
            0.0 if not np.isfinite(pssm_auroc) else pssm_auroc,
            random_model_metrics["cnn"]["auroc"],
            random_model_metrics["transformer"]["auroc"],
            0.0 if not np.isfinite(best_held_mean) else best_held_mean,
        ],
        "AUROC",
        "Binder AUROC",
        results_dir / "auroc_comparison.png",
    )
    plot_pred_vs_measured(y[test_idx], best_test, results_dir / "pred_vs_measured.png", seed=cfg.seed)
    plot_learning_curves(
        {"cnn": cnn_hist.val_auroc, "transformer": transformer_hist.val_auroc},
        results_dir / "learning_curves.png",
    )
    finite_held = [(allele, score) for allele, score in best_held_per.items() if np.isfinite(score)]
    if finite_held:
        plot_per_allele(
            [name for name, _ in finite_held],
            [score for _, score in finite_held],
            results_dir / "heldout_per_allele_auroc.png",
            "Held-out allele AUROC",
        )

    top = ranked.iloc[0] if len(ranked) else None
    metrics = {
        "project": "hla-peptide-binding",
        "success_criteria": [
            criterion(CRITERION_AUROC, random_model_metrics[best_name]["auroc"], ">=", 0.90),
            criterion(CRITERION_SPEARMAN, random_model_metrics[best_name]["spearman"], ">=", 0.70),
            criterion(CRITERION_HELDOUT, best_held_mean, ">=", 0.80),
            criterion(CRITERION_BASELINE, gap, ">=", 0.03),
        ],
        "metrics": {
            "n_pairs": int(len(frame)),
            "n_heldout_alleles": int(len(heldout_alleles)),
            "binder_rate": float(binders.mean()),
            "random_test": {
                "cnn": random_model_metrics["cnn"],
                "transformer": random_model_metrics["transformer"],
                "best_model": best_name,
                "best": {
                    "auroc": random_model_metrics[best_name]["auroc"],
                    "spearman": random_model_metrics[best_name]["spearman"],
                },
                "selection_val_auroc": {name: float(val_choice[name][0]) for name in val_choice},
            },
            "heldout_alleles": {
                "mean_per_allele_auroc": best_held_mean,
                "cnn_mean_per_allele_auroc": held_means["cnn"],
                "transformer_mean_per_allele_auroc": held_means["transformer"],
                "per_allele_auroc": best_held_per,
                "n_alleles_with_auroc": int(sum(np.isfinite(v) for v in best_held_per.values())),
                "pssm": "PSSM cannot score unseen alleles",
            },
            "neoantigen": {
                "protein": NY_ESO1_NAME,
                "uniprot": NY_ESO1_UNIPROT,
                "allele": demo_allele,
                "allele_in_training": demo_allele in set(train_alleles),
                "n_ninemers": int(len(ranked)),
                "top_peptide": None if top is None else str(top["peptide"]),
                "top_score": None if top is None else float(top["model_score"]),
                "known_epitope": KNOWN_A0201_EPITOPE,
                "known_epitope_rank": epitope_rank,
                "known_epitope_score": epitope_score,
            },
        },
        "baseline": {
            "name": "per-allele PSSM",
            "random_test_auroc": pssm_auroc,
            "random_test_spearman": pssm_spearman,
            "n_scored_random": int(shared.sum()),
            "best_model_auroc_on_pssm_rows": model_on_shared,
            "auroc_gap_best_minus_pssm": gap,
            "heldout": "PSSM cannot score unseen alleles",
        },
        "splits": {
            "random": {
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            },
            "heldout_alleles": {"train": 0, "val": 0, "test": int(len(held_idx))},
        },
        "seeds": [cfg.seed],
        "hardware": {"cpu": None, "ram_gb": None, "gpu": None, "python": None, "os": None, "peak_rss_gb": None},
        "runtime_seconds": None,
    }
    write_json(results_dir / "metrics.json", metrics)
    return metrics


def _assert_split_integrity(
    alleles: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    held_idx: np.ndarray,
    heldout_alleles: list[str],
) -> None:
    groups = [train_idx, val_idx, test_idx, held_idx]
    seen: set[int] = set()
    for group in groups:
        ids = set(int(i) for i in group.tolist())
        if seen & ids:
            raise RuntimeError("Split indices overlap.")
        seen |= ids
    if seen != set(range(len(alleles))):
        raise RuntimeError("Split indices do not cover every row.")
    held = set(heldout_alleles)
    for group in (train_idx, val_idx, test_idx):
        present = set(alleles[group].astype(str).tolist())
        if present & held:
            raise RuntimeError("A held-out allele leaked into train, val, or the random test split.")
    if set(alleles[held_idx].astype(str).tolist()) != held:
        raise RuntimeError("Held-out rows do not match the held-out allele list.")
