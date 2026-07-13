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

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec

if TYPE_CHECKING:
    from collections.abc import Mapping

OUTPUT_LAYOUT_VERSION = 2
STATE_FILENAME = "eai_output_state.json"
SUCCESS_FILENAME = "_SUCCESS"


def manifest_identity(path: str | Path) -> tuple[str, int]:
    """Return the byte-level SHA-256 and usable key count for a manifest."""
    digest = hashlib.sha256()
    count = 0
    with open(path, "rb") as fh:
        for raw in fh:
            digest.update(raw)
            line = raw.strip()
            if line and not line.startswith(b"#"):
                count += 1
    return digest.hexdigest(), count


def success_marker_path(output_dir: str) -> str:
    return f"{output_dir.rstrip('/')}/{SUCCESS_FILENAME}"


def _url_to_fs(path: str, storage_options: dict[str, Any] | None) -> tuple[Any, str]:
    return fsspec.core.url_to_fs(path, **(storage_options or {}))


def remove_output(path: str, storage_options: dict[str, Any] | None = None) -> None:
    """Remove an output prefix recursively if it exists."""
    fs, fs_path = _url_to_fs(path, storage_options)
    if fs.exists(fs_path):
        fs.rm(fs_path, recursive=True)


def write_json(path: str, payload: Mapping[str, Any], storage_options: dict[str, Any] | None = None) -> None:
    """Write JSON atomically for local files and as one object PUT remotely."""
    encoded = (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode()
    fs, fs_path = _url_to_fs(path, storage_options)
    if "://" in path:
        with fs.open(fs_path, "wb") as fh:
            fh.write(encoded)
        return

    target = Path(fs_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as fh:
        tmp_path = fh.name
        fh.write(encoded)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, target)


def initialize_output(
    *,
    checkpoint_path: str | Path,
    manifest_sha256: str,
    pdf_output_dir: str,
    cdx_output_dir: str | None,
    storage_options: dict[str, Any] | None,
) -> bool:
    """Purge legacy output exactly once, then persist the resumable layout.

    Returns True when this call performed the migration, False on a matching
    resumed attempt. A changed manifest/layout is rejected rather than mixed.
    """
    checkpoint = Path(checkpoint_path).absolute()
    checkpoint.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint / STATE_FILENAME
    expected = {
        "layout_version": OUTPUT_LAYOUT_VERSION,
        "manifest_sha256": manifest_sha256,
        "pdf_output_dir": pdf_output_dir.rstrip("/") + "/",
        "cdx_output_dir": cdx_output_dir.rstrip("/") + "/" if cdx_output_dir else None,
    }
    lock_path = checkpoint / f".{STATE_FILENAME}.lock"
    with open(lock_path, "a+b") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if state_path.is_file():
            actual = json.loads(state_path.read_text())
            if actual != expected:
                msg = (
                    f"Checkpoint/output identity mismatch at {state_path}: "
                    "use a fresh checkpoint or explicitly reset the full run."
                )
                raise ValueError(msg)
            return False

        metadata_dir = checkpoint / ".nemo_curator_metadata"
        if metadata_dir.exists():
            msg = (
                f"Native checkpoint metadata exists without {STATE_FILENAME} at {checkpoint}; "
                "refusing to purge outputs that may already be checkpointed."
            )
            raise RuntimeError(msg)

        remove_output(pdf_output_dir, storage_options)
        if cdx_output_dir:
            remove_output(cdx_output_dir, storage_options)
        write_json(str(state_path), expected)
        return True
