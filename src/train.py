"""CPU training loop. Model choice uses validation AUROC only."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.metrics import safe_auroc, safe_spearman


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def limit_torch_threads(n: int = 4) -> None:
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class _PairDataset(Dataset):
    def __init__(self, pep: np.ndarray, mhc: np.ndarray, target: np.ndarray) -> None:
        self.pep = pep
        self.mhc = mhc
        self.target = target.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.pep[index]),
            torch.from_numpy(self.mhc[index]),
            torch.tensor(self.target[index], dtype=torch.float32),
        )


@dataclass
class TrainHistory:
    train_loss: list[float]
    val_loss: list[float]
    val_auroc: list[float]
    val_spearman: list[float]
    best_epoch: int
    best_val_auroc: float


def predict(
    model: nn.Module,
    pep: np.ndarray,
    mhc: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Predicted normalized affinity, higher means stronger binding."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pep), batch_size):
            pep_b = torch.from_numpy(pep[start : start + batch_size]).to(device)
            mhc_b = torch.from_numpy(mhc[start : start + batch_size]).to(device)
            pred = model(pep_b, mhc_b).detach().cpu().numpy()
            outputs.append(pred)
    if not outputs:
        return np.array([], dtype=np.float64)
    return np.concatenate(outputs).astype(np.float64, copy=False)


def train_regressor(
    model: nn.Module,
    train_pep: np.ndarray,
    train_mhc: np.ndarray,
    train_y: np.ndarray,
    train_binder: np.ndarray,
    val_pep: np.ndarray,
    val_mhc: np.ndarray,
    val_y: np.ndarray,
    val_binder: np.ndarray,
    *,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> TrainHistory:
    """Minimize MSE on the normalized affinity. Early-stop on validation AUROC.

    ``train_binder`` is unused during the update. It is accepted so the call
    site stays explicit that labels for AUROC are not the training target.
    """
    del train_binder  # selection uses validation binders, not the training loss
    limit_torch_threads(4)
    model.to(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        _PairDataset(train_pep, train_mhc, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_auroc = -1.0
    best_epoch = 0
    waits = 0
    history = TrainHistory([], [], [], [], 0, float("nan"))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for pep_b, mhc_b, y_b in loader:
            pep_b = pep_b.to(device)
            mhc_b = mhc_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(pep_b, mhc_b)
            loss = nn.functional.mse_loss(pred, y_b)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(y_b.shape[0])
            total_n += int(y_b.shape[0])
        train_loss = total_loss / max(total_n, 1)
        val_pred = predict(model, val_pep, val_mhc, device, batch_size=max(batch_size, 1024))
        val_loss = float(np.mean((val_pred - val_y) ** 2)) if len(val_y) else float("nan")
        val_auroc = safe_auroc(val_binder, val_pred)
        val_spearman = safe_spearman(val_y, val_pred)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_auroc.append(val_auroc)
        history.val_spearman.append(val_spearman)
        improved = np.isfinite(val_auroc) and val_auroc > best_auroc + 1e-4
        if improved:
            best_auroc = float(val_auroc)
            best_epoch = epoch
            waits = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waits += 1
        print(
            f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_auroc={val_auroc:.4f} val_spearman={val_spearman:.4f}",
            flush=True,
        )
        if waits >= patience:
            print(f"early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no finite validation AUROC.")
    model.load_state_dict(best_state)
    history.best_epoch = best_epoch
    history.best_val_auroc = best_auroc
    return history
