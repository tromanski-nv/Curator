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

from typing import Any, Literal

import cudf

from nemo_curator.stages.deduplication.exact.identification import ExactDuplicateIdentification
from nemo_curator.stages.interleaved.utils.deduplication import sample_ordering

InterleavedTextMode = Literal["metadata_content", "text_rows"]


class InterleavedExactDuplicateIdentification(ExactDuplicateIdentification):
    """Find exact duplicates after reconstructing one text document per interleaved sample."""

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        text_mode: InterleavedTextMode = "text_rows",
        text_field: str = "text",
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        total_nparts: int | None = None,
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        enable_statistics: bool = False,
        sample_id_field: str = "sample_id",
        position_field: str = "position",
        modality_field: str = "modality",
        text_content_field: str = "text_content",
        text_modality: str = "text",
        metadata_modality: str = "metadata",
        metadata_json_path: str | None = "$.content",
        text_separator: str = "\n\n",
    ):
        if text_mode not in ("metadata_content", "text_rows"):
            msg = "text_mode must be one of {'metadata_content', 'text_rows'}"
            raise ValueError(msg)
        super().__init__(
            text_field=text_field,
            output_path=output_path,
            input_filetype="parquet",
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
            assign_id=True,
            total_nparts=total_nparts,
            rmm_pool_size=rmm_pool_size,
            spill_memory_limit=spill_memory_limit,
            enable_statistics=enable_statistics,
        )
        self.text_mode = text_mode
        self.sample_id_field = sample_id_field
        self.position_field = position_field
        self.modality_field = modality_field
        self.text_content_field = text_content_field
        self.text_modality = text_modality
        self.metadata_modality = metadata_modality
        self.metadata_json_path = metadata_json_path
        self.text_separator = text_separator

    def _extract_metadata_content(self, df: cudf.DataFrame) -> cudf.DataFrame:
        rows = df[df[self.modality_field] == self.metadata_modality][[self.sample_id_field, self.text_content_field]]
        row_count_field = "_metadata_row_count"
        rows[row_count_field] = 1
        counts = rows.groupby(self.sample_id_field, sort=True).agg({row_count_field: "sum"}).reset_index()
        duplicate_samples = counts[counts[row_count_field] > 1]
        if len(duplicate_samples) > 0:
            examples = duplicate_samples[self.sample_id_field].head().to_arrow().to_pylist()
            msg = f"Found multiple metadata rows for samples such as {examples}"
            raise ValueError(msg)

        if len(rows) == 0:
            return cudf.DataFrame(
                {
                    self.sample_id_field: cudf.Series([], dtype=df[self.sample_id_field].dtype),
                    self.text_field: cudf.Series([], dtype="str"),
                },
            )
        if self.metadata_json_path is None:
            rows[self.text_field] = rows[self.text_content_field]
        else:
            rows[self.text_field] = rows[self.text_content_field].str.get_json_object(self.metadata_json_path)
        return rows[[self.sample_id_field, self.text_field]]

    def _extract_text_rows(self, df: cudf.DataFrame) -> cudf.DataFrame:
        rows = df[df[self.modality_field] == self.text_modality][
            [self.sample_id_field, self.position_field, self.text_content_field]
        ]
        rows = rows[rows[self.text_content_field].notnull()]
        if len(rows) == 0:
            return cudf.DataFrame(
                {
                    self.sample_id_field: cudf.Series([], dtype=df[self.sample_id_field].dtype),
                    self.text_field: cudf.Series([], dtype="str"),
                },
            )
        rows = rows.sort_values([self.sample_id_field, self.position_field])
        rows = rows.groupby(self.sample_id_field, sort=True).agg({self.text_content_field: list})
        rows[self.text_field] = rows[self.text_content_field].str.join(self.text_separator)
        return rows.reset_index()[[self.sample_id_field, self.text_field]]

    def _read_files(self, filepaths: list[str]) -> cudf.DataFrame:
        read_kwargs = self.read_kwargs.copy()
        columns_override = read_kwargs.pop("columns", None)
        if columns_override is not None:
            msg = "Columns cannot be set in read_kwargs for interleaved exact deduplication"
            raise ValueError(msg)
        columns = [self.sample_id_field, self.modality_field, self.text_content_field]
        if self.text_mode == "text_rows":
            columns.append(self.position_field)
        df = self.read_parquet(filepaths, columns=columns, assign_id=False, **read_kwargs)

        documents = sample_ordering(df, self.sample_id_field).to_frame(name=self.sample_id_field)
        if len(documents) == 0:
            msg = "No interleaved samples found"
            raise ValueError(msg)
        extracted_text = (
            self._extract_text_rows(df) if self.text_mode == "text_rows" else self._extract_metadata_content(df)
        )
        documents = documents.merge(extracted_text, on=self.sample_id_field, how="left")
        documents = documents.sort_values(self.sample_id_field).reset_index(drop=True)
        documents = self.assign_id(filepaths, documents)
        return documents[documents[self.text_field].notnull()][[self.id_field, self.text_field]]
