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
from typing import TYPE_CHECKING, Any

import pandas as pd

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.deduplication.id_generator import CURATOR_DEDUP_ID_STR
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.utils.file_utils import get_fs

if TYPE_CHECKING:
    from nemo_curator.backends.base import WorkerMetadata


@dataclass
class InterleavedSampleDuplicatesRemovalStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Remove complete interleaved samples whose deduplication IDs were marked as duplicates."""

    ids_to_remove_path: str
    id_field: str = CURATOR_DEDUP_ID_STR
    duplicate_id_field: str = CURATOR_DEDUP_ID_STR
    sample_id_field: str = "sample_id"
    read_kwargs: dict[str, Any] | None = None
    drop_id_field: bool = True
    name: str = "interleaved_sample_duplicates_removal"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = self.read_kwargs.copy() if self.read_kwargs else {}
        self._removal_fs = get_fs(self.ids_to_remove_path, self.read_kwargs.get("storage_options", {}))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        output_fields = [self.sample_id_field]
        if not self.drop_id_field:
            output_fields.append(self.id_field)
        return ["data"], output_fields

    def _has_removal_files(self) -> bool:
        if not self._removal_fs.exists(self.ids_to_remove_path):
            return False
        if self._removal_fs.isdir(self.ids_to_remove_path):
            return any(path.endswith(".parquet") for path in self._removal_fs.find(self.ids_to_remove_path))
        return True

    def _read_removal_subset(self, min_id: int, max_id: int) -> pd.DataFrame:
        if not self._has_removal_files():
            return pd.DataFrame({self.duplicate_id_field: pd.Series(dtype="int64")})
        return pd.read_parquet(
            self.ids_to_remove_path,
            filters=[(self.duplicate_id_field, ">=", min_id), (self.duplicate_id_field, "<=", max_id)],
            columns=[self.duplicate_id_field],
            **self.read_kwargs,
        )

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas()
        if self.id_field not in df.columns:
            msg = f"Input interleaved batch is missing required ID field {self.id_field!r}"
            raise ValueError(msg)
        if self.sample_id_field not in df.columns:
            msg = f"Input interleaved batch is missing required sample field {self.sample_id_field!r}"
            raise ValueError(msg)
        if len(df) == 0:
            if self.drop_id_field:
                df = df.drop(columns=[self.id_field])
            return InterleavedBatch(
                dataset_name=task.dataset_name,
                data=df,
                _metadata={
                    **task._metadata,
                    "num_removed": 0,
                    "num_samples_in": 0,
                    "num_samples_removed": 0,
                    "num_samples_out": 0,
                    "num_rows_in": 0,
                    "num_rows_out": 0,
                },
                _stage_perf=task._stage_perf,
            )

        started = time.perf_counter()
        min_id = int(df[self.id_field].min())
        max_id = int(df[self.id_field].max())
        min_max_time = time.perf_counter() - started

        started = time.perf_counter()
        removal_df = self._read_removal_subset(min_id, max_id)
        read_dupes_time = time.perf_counter() - started

        started = time.perf_counter()
        removal_ids = set(removal_df[self.duplicate_id_field].tolist())
        duplicate_rows = df[self.id_field].isin(removal_ids)
        sample_ids_to_drop = set(df.loc[duplicate_rows, self.sample_id_field].tolist())
        num_samples_in = int(df[self.sample_id_field].nunique())
        df_kept = df[~df[self.sample_id_field].isin(sample_ids_to_drop)]
        removal_time = time.perf_counter() - started

        if self.drop_id_field:
            df_kept = df_kept.drop(columns=[self.id_field])

        self._log_metrics(
            {
                "input_df_min_max_time": min_max_time,
                "read_dupes_time": read_dupes_time,
                "id_removal_time": removal_time,
            },
        )
        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=df_kept.reset_index(drop=True),
            _metadata={
                **task._metadata,
                "num_removed": len(sample_ids_to_drop),
                "num_samples_in": num_samples_in,
                "num_samples_removed": len(sample_ids_to_drop),
                "num_samples_out": num_samples_in - len(sample_ids_to_drop),
                "num_rows_in": len(df),
                "num_rows_out": len(df_kept),
            },
            _stage_perf=task._stage_perf,
        )


@dataclass
class InterleavedSampleIdRemovalStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """Remove complete interleaved samples using a persisted string sample-ID set."""

    ids_to_remove_path: str
    sample_id_field: str = "sample_id"
    duplicate_id_field: str = "sample_id"
    read_kwargs: dict[str, Any] | None = None
    name: str = "interleaved_sample_id_removal"

    def __post_init__(self) -> None:
        super().__init__()
        self.read_kwargs = self.read_kwargs.copy() if self.read_kwargs else {}
        self._removal_fs = get_fs(self.ids_to_remove_path, self.read_kwargs.get("storage_options", {}))
        self._sample_ids_to_remove: set[str] | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], [self.sample_id_field]

    def setup(self, _worker_metadata: WorkerMetadata | None = None) -> None:
        if not self._removal_fs.exists(self.ids_to_remove_path):
            self._sample_ids_to_remove = set()
            return
        removal_df = pd.read_parquet(
            self.ids_to_remove_path,
            columns=[self.duplicate_id_field],
            **self.read_kwargs,
        )
        self._sample_ids_to_remove = set(removal_df[self.duplicate_id_field].dropna().astype(str))

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        if self._sample_ids_to_remove is None:
            self.setup()
        df = task.to_pandas()
        if self.sample_id_field not in df.columns:
            msg = f"Input interleaved batch is missing required sample field {self.sample_id_field!r}"
            raise ValueError(msg)
        samples_in = int(df[self.sample_id_field].nunique())
        rows_to_drop = df[self.sample_id_field].astype(str).isin(self._sample_ids_to_remove)
        removed_sample_ids = set(df.loc[rows_to_drop, self.sample_id_field].astype(str))
        df_kept = df.loc[~rows_to_drop].reset_index(drop=True)
        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=df_kept,
            _metadata={
                **task._metadata,
                "num_samples_in": samples_in,
                "num_samples_removed": len(removed_sample_ids),
                "num_samples_out": samples_in - len(removed_sample_ids),
                "num_rows_in": len(df),
                "num_rows_out": len(df_kept),
            },
            _stage_perf=task._stage_perf,
        )
