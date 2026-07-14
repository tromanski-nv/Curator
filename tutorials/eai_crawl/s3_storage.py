# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S3 / SwiftStack storage helpers for EAI crawl outputs.

Input WARCs (``team-vendor-data``) and output indexes (``eai-data``) often use
**different** credentials on the same endpoint. Pass explicit ``storage_options``
for writes so they do not collide with ``AWS_*`` used for reads.
"""

from __future__ import annotations

import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any

import pandas as pd


def is_remote_url(path: str) -> bool:
    return "://" in path


def rclone_s3_storage_options(remote: str = "eai-data") -> dict[str, Any]:
    """Build s3fs/fsspec ``storage_options`` from an rclone S3 remote.

    Reads ``~/.config/rclone/rclone.conf`` (never prints secrets).
    """
    cfg_path = Path.home() / ".config/rclone/rclone.conf"
    cfg = ConfigParser()
    if not cfg_path.is_file() or not cfg.read(cfg_path) or remote not in cfg:
        msg = f"rclone remote [{remote}] not found in {cfg_path}"
        raise FileNotFoundError(msg)

    sec = cfg[remote]
    endpoint = sec.get("endpoint") or os.environ.get("AWS_ENDPOINT_URL") or None
    options: dict[str, Any] = {
        "key": sec["access_key_id"],
        "secret": sec["secret_access_key"],
        "client_kwargs": {
            "endpoint_url": endpoint,
            "region_name": sec.get("region") or "us-east-1",
        },
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }
    if sec.get("session_token"):
        options["token"] = sec["session_token"]
    return options


def env_s3_storage_options(prefix: str = "EAI_OUT_") -> dict[str, Any] | None:
    """Build storage_options from ``{prefix}AWS_ACCESS_KEY_ID`` etc., if set."""
    key = os.environ.get(f"{prefix}AWS_ACCESS_KEY_ID")
    secret = os.environ.get(f"{prefix}AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        return None
    endpoint = os.environ.get(f"{prefix}AWS_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")
    region = os.environ.get(f"{prefix}AWS_DEFAULT_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    options: dict[str, Any] = {
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint, "region_name": region},
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }
    token = os.environ.get(f"{prefix}AWS_SESSION_TOKEN")
    if token:
        options["token"] = token
    return options


def resolve_output_storage_options(
    *,
    rclone_remote: str | None = None,
    explicit: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Prefer explicit options, then ``EAI_OUT_*`` env, then rclone remote."""
    if explicit:
        return explicit
    from_env = env_s3_storage_options()
    if from_env:
        return from_env
    if rclone_remote:
        return rclone_s3_storage_options(rclone_remote)
    return None


def ensure_parent(path: str, storage_options: dict[str, Any] | None = None) -> None:
    """Create parent directory for local paths; no-op for remote URLs."""
    if is_remote_url(path):
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def write_parquet(
    df: pd.DataFrame,
    path: str,
    *,
    storage_options: dict[str, Any] | None = None,
) -> None:
    """Write a DataFrame to local or ``s3://`` Parquet."""
    if is_remote_url(path):
        if not storage_options:
            msg = f"storage_options required for remote path: {path}"
            raise ValueError(msg)
        # Ensure parent "directory" exists as a no-op for S3; s3fs creates on write.
        df.to_parquet(path, index=False, storage_options=storage_options)
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
