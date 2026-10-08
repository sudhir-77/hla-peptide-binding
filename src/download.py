"""Download the TDC MHC-I table and check its checksum and row count."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.affinity import binder_labels
from src.constants import (
    COL_ALLELE,
    COL_ID,
    COL_MHC,
    COL_PEPTIDE,
    COL_TARGET,
    DATA_FILENAME,
    DATA_URL,
    EXPECTED_ROWS,
    EXPECTED_SHA256,
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_data_rows(path: Path) -> int:
    """Count data rows in a headered text table without loading it all at once."""
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            if line.strip():
                n += 1
    return n


def verify_dataset(path: Path) -> None:
    """Raise if the cached file is not the expected TDC release."""
    digest = sha256_file(path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: got {digest}, expected {EXPECTED_SHA256}."
        )
    rows = count_data_rows(path)
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Row count mismatch for {path.name}: got {rows}, expected {EXPECTED_ROWS}.")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    headers = {"User-Agent": "hla-peptide-binding/1.0"}
    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 16):
                        if block:
                            handle.write(block)
            tmp.replace(dest)
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Failed to download {url}") from last_error


def ensure_dataset(data_dir: Path) -> Path:
    """Return the cached table path, downloading and verifying it if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / DATA_FILENAME
    if path.exists():
        try:
            verify_dataset(path)
            return path
        except RuntimeError:
            path.unlink()
    _download(DATA_URL, path)
    verify_dataset(path)
    return path


def load_binding_table(path: Path) -> pd.DataFrame:
    """Load the TDC table into the columns the rest of the pipeline uses.

    ``log_affinity`` is the published Y score, already
    ``1 - log10(IC50 nM) / log10(50000)``. Higher means stronger binding.
    """
    raw = pd.read_csv(
        path,
        sep="\t",
        dtype={COL_ID: "int64", COL_PEPTIDE: "string", COL_ALLELE: "string", COL_MHC: "string"},
    )
    missing = {COL_ID, COL_PEPTIDE, COL_ALLELE, COL_MHC, COL_TARGET} - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    log_affinity = pd.to_numeric(raw[COL_TARGET], errors="coerce").to_numpy(dtype=np.float64)
    frame = pd.DataFrame(
        {
            "id": raw[COL_ID].to_numpy(dtype=np.int64),
            "peptide": raw[COL_PEPTIDE].astype(str).str.strip().str.upper(),
            "allele": raw[COL_ALLELE].astype(str).str.strip(),
            "mhc": raw[COL_MHC].astype(str).str.strip().str.upper(),
            "log_affinity": log_affinity,
        }
    )
    frame = frame.dropna(subset=["peptide", "allele", "mhc", "log_affinity"]).reset_index(drop=True)
    frame = frame.loc[frame["peptide"].str.len() > 0].reset_index(drop=True)
    frame["binder"] = binder_labels(frame["log_affinity"].to_numpy())
    return frame
