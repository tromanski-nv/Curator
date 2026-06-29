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

from dataclasses import dataclass
from typing import Any

from loguru import logger

from nemo_curator.stages.text.download import URLGenerator


@dataclass
class S3WarcUrlGenerator(URLGenerator):
    """List WARC object keys under an S3 prefix.

    Returns plain object keys (not ``s3://`` URLs); the downstream
    ``S3WarcMetadataStage`` pairs each key with the bucket for range reads.
    Credentials are resolved by boto3's default chain unless ``client`` is
    injected (useful for tests).
    """

    bucket: str
    prefix: str = ""
    suffix: str = ".warc"  # matches both ".warc" and ".warc.gz"
    limit: int | None = None
    client: Any = None

    def _get_client(self) -> Any:  # noqa: ANN401 - boto3 S3 client has no type stubs
        if self.client is not None:
            return self.client
        try:
            import boto3
        except ModuleNotFoundError as exc:
            msg = "boto3 is required for S3WarcUrlGenerator. Install with: pip install boto3"
            raise RuntimeError(msg) from exc
        return boto3.client("s3")

    def generate_urls(self) -> list[str]:
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
