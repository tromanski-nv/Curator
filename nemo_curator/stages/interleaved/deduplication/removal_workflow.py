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

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.pipeline.workflow import WorkflowBase, WorkflowRunResult
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR

from .removal import InterleavedSampleDuplicatesRemovalStage

if TYPE_CHECKING:
    from nemo_curator.backends.base import BaseExecutor
    from nemo_curator.stages.base import ProcessingStage
    from nemo_curator.tasks import FileGroupTask


@dataclass
class InterleavedDuplicatesRemovalWorkflow(WorkflowBase):
    input_path: str | list[str] | None
    ids_to_remove_path: str
    output_path: str

    input_files_per_partition: int | None = None
    input_blocksize: int | str | None = None
    input_file_extensions: list[str] | None = None
    input_task_limit: int | None = None
    input_kwargs: dict[str, Any] | None = None

    id_field: str = CURATOR_DEDUP_ID_STR
    duplicate_id_field: str = CURATOR_DEDUP_ID_STR
    sample_id_field: str = "sample_id"
    duplicate_id_read_kwargs: dict[str, Any] | None = None

    id_generator_path: str | None = None
    id_generator_storage_options: dict[str, Any] | None = None
    drop_id_field: bool = True

    output_kwargs: dict[str, Any] | None = None
    output_mode: Literal["ignore", "overwrite", "append", "error"] | None = None
    materialize_on_write: bool = False

    def _generate_stages(self, initial_tasks: list[FileGroupTask] | None = None) -> list[ProcessingStage]:
        stages: list[ProcessingStage] = []

        if initial_tasks is None:
            if self.input_path is None:
                msg = "input_path is required when initial_tasks is None"
                raise ValueError(msg)
            from nemo_curator.stages.file_partitioning import FilePartitioningStage

            stages.append(
                FilePartitioningStage(
                    file_paths=self.input_path,
                    files_per_partition=self.input_files_per_partition,
                    blocksize=None if self.input_files_per_partition is not None else self.input_blocksize,
                    file_extensions=self.input_file_extensions,
                    storage_options=(self.input_kwargs or {}).get("storage_options"),
                    limit=self.input_task_limit,
                ),
            )
        else:
            logger.warning(
                "Initial tasks provided; input_path, input_files_per_partition, input_blocksize, "
                "and input_file_extensions are ignored",
            )

        from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
        from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage

        stages.extend(
            [
                InterleavedParquetReaderStage(
                    read_kwargs=self.input_kwargs or {},
                    _assign_ids=self.id_generator_path is not None,
                ),
                InterleavedSampleDuplicatesRemovalStage(
                    ids_to_remove_path=self.ids_to_remove_path,
                    id_field=self.id_field,
                    duplicate_id_field=self.duplicate_id_field,
                    sample_id_field=self.sample_id_field,
                    read_kwargs=self.duplicate_id_read_kwargs,
                    drop_id_field=self.drop_id_field,
                ),
                InterleavedParquetWriterStage(
                    path=self.output_path,
                    write_kwargs=self.output_kwargs or {},
                    materialize_on_write=self.materialize_on_write,
                    **({"mode": self.output_mode} if self.output_mode else {}),
                ),
            ],
        )
        return stages

    @staticmethod
    def _count_removed_duplicates(tasks: list[FileGroupTask] | None) -> int:
        return sum(
            (getattr(task, "_metadata", {}) or {}).get(
                "num_samples_removed",
                (getattr(task, "_metadata", {}) or {}).get("num_removed", 0),
            )
            for task in tasks or []
        )

    def run(
        self,
        executor: BaseExecutor | None = None,
        initial_tasks: list[FileGroupTask] | None = None,
    ) -> WorkflowRunResult:
        pipeline = Pipeline(
            name="interleaved_duplicates_removal_workflow",
            description="Interleaved duplicates removal workflow",
            stages=self._generate_stages(initial_tasks),
        )
        workflow_result = WorkflowRunResult(workflow_name="interleaved_duplicates_removal")
        if (
            self.input_task_limit is not None
            and initial_tasks is not None
            and len(initial_tasks) > self.input_task_limit
        ):
            logger.warning(
                "Truncating {} provided initial tasks to input_task_limit={}",
                len(initial_tasks),
                self.input_task_limit,
            )
            initial_tasks = initial_tasks[: self.input_task_limit]

        if executor is None:
            from nemo_curator.backends.xenna import XennaExecutor

            executor = XennaExecutor()

        if self.id_generator_path is not None:
            from nemo_curator.stages.deduplication.id_generator import (
                create_id_generator_actor,
                kill_id_generator_actor,
            )

            create_id_generator_actor(self.id_generator_path, storage_options=self.id_generator_storage_options)
            try:
                started = time.time()
                output_tasks = pipeline.run(executor, initial_tasks=initial_tasks)
            finally:
                kill_id_generator_actor()
        else:
            started = time.time()
            output_tasks = pipeline.run(executor, initial_tasks=initial_tasks)

        workflow_result.add_pipeline_tasks("removal", output_tasks)
        workflow_result.add_metadata("total_time", time.time() - started)
        workflow_result.add_metadata("num_duplicates_removed", self._count_removed_duplicates(output_tasks))
        return workflow_result
