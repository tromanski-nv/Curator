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
    # Explicit object keys (bypass listing). Used by the byte-chunk driver so a
    # single job can process an arbitrary set of WARCs spanning multiple days.
    keys: list[str] | None = None

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
        # Explicit-keys mode: trust the provided manifest (already filtered/sorted
        # by the driver); skip the S3 LIST entirely.
        if self.keys is not None:
            keys = list(self.keys)
            if self.limit:
                keys = keys[: self.limit]
            logger.info(f"Using {len(keys)} explicit WARC key(s) for s3://{self.bucket}")
            return keys

        client = self._get_client()
        keys = []
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
