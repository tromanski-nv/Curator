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

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.text.download import URLGenerator


@dataclass
class S3WarcUrlGenerator(URLGenerator):
    """List WARC object keys under an S3 (or S3-compatible) prefix.

    Returns plain object keys (not ``s3://`` URLs); the downstream stage pairs
    each key with the bucket to read the object. Credentials are resolved by
    boto3's default chain (``AWS_*`` env vars, ``~/.aws/``) unless ``client`` is
    injected (useful for tests).

    For S3-compatible object stores (SwiftStack, MinIO, Ceph, ...), set
    ``endpoint_url`` (or the ``AWS_ENDPOINT_URL`` env var). Example SwiftStack:
    ``endpoint_url="https://pdx.s8k.io"``.
    """

    bucket: str
    prefix: str = ""
    suffix: str = ".warc"  # matches both ".warc" and ".warc.gz"
    limit: int | None = None
    endpoint_url: str | None = None
    region: str | None = None
    client: Any = None
    key_file: str | None = None

    def _read_key_file(self) -> list[str]:
        """Read an exact, launcher-generated WARC worklist."""
        if self.limit is not None:
            msg = "key_file cannot be combined with limit; each worklist is an exact array shard"
            raise ValueError(msg)

        path = Path(self.key_file or "")
        if not path.is_file():
            msg = f"WARC key file does not exist: {path}"
            raise ValueError(msg)

        keys = path.read_text(encoding="utf-8").splitlines()
        if not keys:
            msg = f"WARC key file is empty: {path}"
            raise ValueError(msg)

        seen: set[str] = set()
        for line_number, key in enumerate(keys, start=1):
            if not key or key != key.strip():
                msg = f"Invalid blank or whitespace-padded key at {path}:{line_number}"
                raise ValueError(msg)
            if self.prefix and not key.startswith(self.prefix):
                msg = f"WARC key escapes prefix {self.prefix!r} at {path}:{line_number}: {key!r}"
                raise ValueError(msg)
            if not key.endswith(self.suffix):
                msg = f"WARC key does not end with {self.suffix!r} at {path}:{line_number}: {key!r}"
                raise ValueError(msg)
            if key in seen:
                msg = f"Duplicate WARC key at {path}:{line_number}: {key!r}"
                raise ValueError(msg)
            seen.add(key)

        logger.info(f"Loaded {len(keys)} WARC object(s) from exact key file {path}")
        return keys

    def _get_client(self) -> Any:  # noqa: ANN401 - boto3 S3 client has no type stubs
        if self.client is not None:
            return self.client
        try:
            import boto3
        except ModuleNotFoundError as exc:
            msg = "boto3 is required for S3WarcUrlGenerator. Install with: pip install boto3"
            raise RuntimeError(msg) from exc
        endpoint = self.endpoint_url or os.environ.get("AWS_ENDPOINT_URL") or None
        from botocore.config import Config as BotoConfig

        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=self.region,
            config=BotoConfig(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )

    def generate_urls(self) -> list[str]:
        if self.key_file is not None:
            return self._read_key_file()

        client = self._get_client()
        keys: list[str] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.suffix in key:
                    keys.append(key)

        keys.sort()
        if self.limit:
            keys = keys[: self.limit]

        logger.info(f"Found {len(keys)} WARC object(s) in s3://{self.bucket}/{self.prefix}")
        return keys
