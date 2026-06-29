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

from collections.abc import Iterator
from typing import Any

from nemo_curator.stages.text.download import DocumentIterator
from tutorials.eai_crawl.pdf_records import PDF_RECORD_COLUMNS, iterate_pdf_warc_records


class EaiWarcIterator(DocumentIterator):
    """Iterate application/pdf WARC responses without reading PDF payloads."""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        yield from iterate_pdf_warc_records(file_path)

    def output_columns(self) -> list[str]:
        return PDF_RECORD_COLUMNS
