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

from typing import Any

from nemo_curator.stages.text.download import DocumentExtractor
from tutorials.eai_crawl.pdf_records import PDF_OUTPUT_COLUMNS, PDF_RECORD_COLUMNS, extract_pdf_record


class EaiPdfExtractor(DocumentExtractor):
    """Map application/pdf WARC records to PDF URLs and metadata (no PDF body)."""

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        return extract_pdf_record(record)

    def input_columns(self) -> list[str]:
        return PDF_RECORD_COLUMNS

    def output_columns(self) -> list[str]:
        return PDF_OUTPUT_COLUMNS
