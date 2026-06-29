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

from pathlib import Path

from loguru import logger

from nemo_curator.stages.text.download import DocumentDownloader


class LocalWarcDownloader(DocumentDownloader):
    """Pass-through "downloader" for WARC files already on a local/shared disk.

    There is nothing to fetch, so ``download()`` simply validates and returns the
    existing path. We deliberately do not copy or hardlink: a copy duplicates the
    data, and even ``os.link`` (a hardlink, which shares the inode without copying
    data) would still create an extra directory entry and only works within a
    single filesystem. ``DocumentDownloadStage`` forwards whatever path we return
    straight to the iterator.
    """

    def _get_output_filename(self, url: str) -> str:
        return Path(url).name

    def _download_to_path(self, url: str, path: str) -> tuple[bool, str | None]:  # noqa: ARG002
        # Unused: download() is overridden to avoid touching local files at all.
        return True, None

    def download(self, url: str) -> str | None:
        source = Path(url).expanduser()
        if not source.is_file():
            logger.error(f"Local WARC file not found: {source}")
            return None
        return str(source)
