# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path

import pytest

from tutorials.eai_crawl.resume import (
    STATE_FILENAME,
    initialize_output,
    manifest_identity,
    success_marker_path,
    write_json,
)


def test_initialize_output_purges_legacy_once(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    pdf = tmp_path / "pdf"
    cdx = tmp_path / "cdx"
    pdf.mkdir()
    cdx.mkdir()
    (pdf / "legacy.parquet").write_bytes(b"partial")
    (cdx / "legacy.parquet").write_bytes(b"partial")

    assert initialize_output(
        checkpoint_path=checkpoint,
        manifest_sha256="abc",
        pdf_output_dir=str(pdf),
        cdx_output_dir=str(cdx),
        storage_options=None,
    )
    assert not pdf.exists()
    assert not cdx.exists()

    pdf.mkdir()
    (pdf / "resumable.parquet").write_bytes(b"keep")
    assert not initialize_output(
        checkpoint_path=checkpoint,
        manifest_sha256="abc",
        pdf_output_dir=str(pdf),
        cdx_output_dir=str(cdx),
        storage_options=None,
    )
    assert (pdf / "resumable.parquet").is_file()


def test_initialize_output_rejects_manifest_change(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    kwargs = {
        "checkpoint_path": checkpoint,
        "pdf_output_dir": str(tmp_path / "pdf"),
        "cdx_output_dir": str(tmp_path / "cdx"),
        "storage_options": None,
    }
    initialize_output(manifest_sha256="first", **kwargs)

    with pytest.raises(ValueError, match="identity mismatch"):
        initialize_output(manifest_sha256="second", **kwargs)

    assert json.loads((checkpoint / STATE_FILENAME).read_text())["manifest_sha256"] == "first"


def test_manifest_identity_hashes_exact_file_and_counts_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "chunk.keys"
    manifest.write_text("# comment\none\n\n two \n")

    digest, count = manifest_identity(manifest)

    assert len(digest) == 64
    assert count == 2


def test_success_marker_is_one_authoritative_json_file(tmp_path: Path) -> None:
    output_dir = str(tmp_path / "pdf")
    marker = success_marker_path(output_dir)

    write_json(marker, {"status": "completed", "manifest_sha256": "abc"})

    assert json.loads((tmp_path / "pdf" / "_SUCCESS").read_text()) == {
        "status": "completed",
        "manifest_sha256": "abc",
    }
