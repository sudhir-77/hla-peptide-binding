from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import DATA_FILENAME, EXPECTED_ROWS, EXPECTED_SHA256
from src.download import (
    count_data_rows,
    ensure_dataset,
    load_binding_table,
    sha256_file,
    verify_dataset,
)


def test_sha_and_row_count_helpers(tmp_path: Path):
    path = tmp_path / "tiny.tab"
    path.write_text("h\na\nb\n", encoding="utf-8")
    assert count_data_rows(path) == 2
    assert len(sha256_file(path)) == 64


def test_verify_rejects_bad_checksum(tmp_path: Path):
    path = tmp_path / "bad.tab"
    path.write_text("only-a-header\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Checksum"):
        verify_dataset(path)


def test_load_binding_table_maps_tdc_columns(tmp_path: Path):
    path = tmp_path / "sample.tab"
    mhc = "A" * 34
    path.write_text(
        "ID1\tX1\tID2\tX2\tY\n"
        f"7\tACDEFGHIK\tHLA-A02:01\t{mhc}\t0.80\n"
        f"8\tCCCCCCCCC\tHLA-B07:02\t{mhc}\t0.10\n",
        encoding="utf-8",
    )
    frame = load_binding_table(path)
    assert list(frame["id"]) == [7, 8]
    assert list(frame["peptide"]) == ["ACDEFGHIK", "CCCCCCCCC"]
    assert list(frame["allele"]) == ["HLA-A02:01", "HLA-B07:02"]
    assert list(frame["binder"]) == [1, 0]
    assert frame["mhc"].str.len().eq(34).all()


def test_load_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.tab"
    path.write_text("A\tB\n1\t2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_binding_table(path)


def test_ensure_dataset_uses_cache_and_can_download(tmp_path: Path, monkeypatch):
    cached = tmp_path / "cache"
    cached.mkdir()
    ready = cached / DATA_FILENAME
    ready.write_bytes(b"cached-bytes")
    monkeypatch.setattr("src.download.verify_dataset", lambda path: None)
    monkeypatch.setattr(
        "src.download.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("download should not run")),
    )
    assert ensure_dataset(cached) == ready

    fresh = tmp_path / "fresh"

    class _Response:
        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size=1):
            yield b"abc"
            yield b"def"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("src.download.requests.get", lambda *args, **kwargs: _Response())
    path = ensure_dataset(fresh)
    assert path.read_bytes() == b"abcdef"


def test_real_dataset_checksum_when_cached():
    path = Path("data") / DATA_FILENAME
    if not path.exists():
        pytest.skip("dataset not downloaded")
    verify_dataset(path)
    frame = load_binding_table(path)
    assert len(frame) == EXPECTED_ROWS
    assert sha256_file(path) == EXPECTED_SHA256
    assert frame["binder"].isin([0, 1]).all()
    assert np.isfinite(frame["log_affinity"]).all()
