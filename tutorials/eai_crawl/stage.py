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

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.download import DocumentDownloadExtractStage
from tutorials.eai_crawl.download import LocalWarcDownloader
from tutorials.eai_crawl.extractor import EaiPdfExtractor
from tutorials.eai_crawl.iterator import EaiWarcIterator
from tutorials.eai_crawl.url_generation import LocalWarcUrlGenerator


class EaiCrawlDownloadExtractStage(DocumentDownloadExtractStage):
    """Load local WARC files and collect PDF URLs from application/pdf responses."""

    def __init__(  # noqa: PLR0913
        self,
        download_dir: str = "./eai_warc_downloads",
        warc_dir: str | None = None,
        warc_paths: list[str] | None = None,
        url_limit: int | None = None,
        record_limit: int | None = None,
        add_filename_column: bool | str = True,
        verbose: bool = False,
    ):
        self.url_generator = LocalWarcUrlGenerator(
            warc_dir=warc_dir,
            warc_paths=warc_paths,
            limit=url_limit,
        )
        self.downloader = LocalWarcDownloader(download_dir=download_dir, verbose=verbose)
        self.iterator = EaiWarcIterator()
        self.extractor = EaiPdfExtractor()

        super().__init__(
            url_generator=self.url_generator,
            downloader=self.downloader,
            iterator=self.iterator,
            extractor=self.extractor,
            url_limit=url_limit,
            record_limit=record_limit,
            add_filename_column=add_filename_column,
        )
        self.name = "eai_crawl_pdf_extract"

    def decompose(self) -> list[ProcessingStage]:
        return self.stages

    def get_description(self) -> str:
        return "Collect PDF URLs and metadata from application/pdf records in local WARC files"
