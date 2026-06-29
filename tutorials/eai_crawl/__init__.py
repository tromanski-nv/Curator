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

__all__ = ["EaiCrawlDownloadExtractStage", "S3EaiCrawlStage", "S3WarcMetadataStage"]


def __getattr__(name: str):  # noqa: ANN202
    if name == "EaiCrawlDownloadExtractStage":
        from tutorials.eai_crawl.stage import EaiCrawlDownloadExtractStage

        return EaiCrawlDownloadExtractStage
    if name in {"S3EaiCrawlStage", "S3WarcMetadataStage"}:
        from tutorials.eai_crawl import s3_stage

        return getattr(s3_stage, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
