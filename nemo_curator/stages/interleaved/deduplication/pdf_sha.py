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

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask


@dataclass
class PdfSha256InventoryStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    """Hash original PDF byte streams referenced by interleaved Parquet samples."""

    output_path: str
    pdf_root: str
    sample_id_field: str = "sample_id"
    pdf_name_field: str = "pdf_name"
    chunk_size_bytes: int = 8 * 1024 * 1024
    validate_sample_pdf_mapping: bool = True
    name: str = "pdf_sha256_inventory"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        super().__init__()
        if self.chunk_size_bytes < 1:
            msg = "chunk_size_bytes must be positive"
            raise ValueError(msg)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    @staticmethod
    def _partition_key(filepaths: list[str]) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, ";".join(filepaths)))

    def _hash_pdf(self, source_path: Path) -> tuple[int | None, str | None, str | None]:
        try:
            digest = hashlib.sha256()
            size_bytes = 0
            with source_path.open("rb") as stream:
                while chunk := stream.read(self.chunk_size_bytes):
                    digest.update(chunk)
                    size_bytes += len(chunk)
            return size_bytes, digest.hexdigest(), None
        except OSError as error:
            return None, None, f"{type(error).__name__}: {error}"

    def _read_sample_inventory(self, filepaths: list[str]) -> pd.DataFrame:
        frames = []
        for filepath in filepaths:
            frame = pq.read_table(filepath, columns=[self.sample_id_field, self.pdf_name_field]).to_pandas()
            frame["source_parquet"] = filepath
            frames.append(frame)
        inventory = pd.concat(frames, ignore_index=True).drop_duplicates()

        sample_pdf_counts = inventory.groupby(self.sample_id_field)[self.pdf_name_field].nunique(dropna=False)
        pdf_sample_counts = inventory.groupby(self.pdf_name_field)[self.sample_id_field].nunique(dropna=False)
        if (sample_pdf_counts != 1).any() or (pdf_sample_counts != 1).any():
            msg = "Expected a one-to-one sample_id to pdf_name mapping within each input file group"
            raise ValueError(msg)
        inventory = inventory.drop_duplicates(subset=[self.sample_id_field], keep="first")
        if self.validate_sample_pdf_mapping:
            expected_ids = inventory[self.pdf_name_field].astype(str).str.removesuffix(".pdf")
            mismatches = inventory[self.sample_id_field].astype(str) != expected_ids
            if mismatches.any():
                examples = (
                    inventory.loc[mismatches, [self.sample_id_field, self.pdf_name_field]].head().to_dict("records")
                )
                msg = f"sample_id does not match pdf_name without the .pdf suffix; examples: {examples}"
                raise ValueError(msg)
        return inventory

    def process(self, task: FileGroupTask) -> FileGroupTask:
        output_file = str(Path(self.output_path) / f"{self._partition_key(task.data)}.parquet")
        if Path(output_file).exists():
            return FileGroupTask(
                dataset_name=f"{task.dataset_name}_pdf_sha256_inventory",
                data=[output_file],
                _metadata={**task._metadata, "resumed": True},
                _stage_perf=task._stage_perf,
            )

        inventory = self._read_sample_inventory(task.data)
        rows = []
        for record in inventory.to_dict("records"):
            source_path = Path(self.pdf_root) / str(record[self.pdf_name_field])
            size_bytes, sha256, hash_error = self._hash_pdf(source_path)
            rows.append(
                {
                    self.sample_id_field: record[self.sample_id_field],
                    self.pdf_name_field: record[self.pdf_name_field],
                    "source_path": str(source_path),
                    "source_parquet": record["source_parquet"],
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                    "hash_error": hash_error,
                },
            )

        output_schema = pa.schema(
            [
                pa.field(self.sample_id_field, pa.string()),
                pa.field(self.pdf_name_field, pa.string()),
                pa.field("source_path", pa.string()),
                pa.field("source_parquet", pa.string()),
                pa.field("size_bytes", pa.int64()),
                pa.field("sha256", pa.string()),
                pa.field("hash_error", pa.string()),
            ],
        )
        output_table = pa.Table.from_pylist(rows, schema=output_schema)
        temporary_file = f"{output_file}.tmp.{os.getpid()}"
        pq.write_table(output_table, temporary_file)
        os.replace(temporary_file, output_file)
        num_errors = sum(row["hash_error"] is not None for row in rows)
        return FileGroupTask(
            dataset_name=f"{task.dataset_name}_pdf_sha256_inventory",
            data=[output_file],
            _metadata={
                **task._metadata,
                "num_samples": len(rows),
                "num_hash_errors": num_errors,
                "resumed": False,
            },
            _stage_perf=task._stage_perf,
        )
