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

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow

from .exact_identification import InterleavedExactDuplicateIdentification, InterleavedTextMode


class InterleavedExactDeduplicationWorkflow(ExactDeduplicationWorkflow):
    """Run existing exact-hash shuffle logic on reconstructed interleaved sample text."""

    def __init__(  # noqa: PLR0913
        self,
        output_path: str,
        input_path: str | list[str] | None = None,
        input_blocksize: str | int = "2GiB",
        identification_batchsize: int = 1,
        input_file_extensions: list[str] | None = None,
        read_kwargs: dict[str, Any] | None = None,
        write_kwargs: dict[str, Any] | None = None,
        text_mode: InterleavedTextMode = "text_rows",
        text_field: str = "text",
        metadata_json_path: str | None = "$.content",
        text_separator: str = "\n\n",
        total_nparts: int | None = None,
        rmm_pool_size: int | Literal["auto"] | None = "auto",
        spill_memory_limit: int | Literal["auto"] | None = "auto",
        env_vars: dict[str, Any] | None = None,
    ):
        super().__init__(
            output_path=output_path,
            input_path=input_path,
            input_filetype="parquet",
            input_blocksize=input_blocksize,
            identification_batchsize=identification_batchsize,
            input_file_extensions=input_file_extensions,
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
            assign_id=True,
            text_field=text_field,
            perform_removal=False,
            total_nparts=total_nparts,
            rmm_pool_size=rmm_pool_size,
            spill_memory_limit=spill_memory_limit,
            env_vars=env_vars,
        )
        self.text_mode = text_mode
        self.metadata_json_path = metadata_json_path
        self.text_separator = text_separator

    def _create_identification_pipeline(self, num_input_tasks: int) -> Pipeline:
        total_nparts = max(1, num_input_tasks // 3) if self.total_nparts is None else max(1, self.total_nparts)
        return Pipeline(
            name="interleaved_exact_deduplication_pipeline",
            stages=[
                InterleavedExactDuplicateIdentification(
                    output_path=self.output_path,
                    text_mode=self.text_mode,
                    text_field=self.text_field,
                    read_kwargs=self.read_kwargs,
                    write_kwargs=self.write_kwargs,
                    total_nparts=total_nparts,
                    rmm_pool_size=self.rmm_pool_size,
                    spill_memory_limit=self.spill_memory_limit,
                    metadata_json_path=self.metadata_json_path,
                    text_separator=self.text_separator,
                ).with_(batch_size=int(self.identification_batchsize)),
            ],
        )
