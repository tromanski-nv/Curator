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

"""Split arXiv source tars into per-submission work units."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass, field
from typing import Any

import fsspec
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask

#: Outer-tar members that hold a LaTeX submission.  ``.pdf`` members are
#: PDF-only submissions with no source and are counted but not emitted.
DEFAULT_SUBMISSION_EXTENSIONS: tuple[str, ...] = (".gz",)


@dataclass
class ArxivTarPartitioningStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    r"""Index arXiv source tars and emit fixed-size groups of submissions.

    An ``arXiv_src_YYMM_NNN.tar`` shard holds ~2400 submissions as
    ``YYMM/<arxiv_id>.gz`` members and is commonly ~500 MB.  Handing a whole
    shard to one worker would serialize ~2400 papers and build a multi-GB Arrow
    table before anything is emitted.  This stage instead reads only the tar's
    member headers -- ~0.35 s for a 227 MB shard, since ``tarfile`` seeks rather
    than reads -- and hands downstream workers byte ranges they can seek to
    directly.

    Each output :class:`FileGroupTask` carries JSON strings in ``data``, one per
    submission, in the same style as ``PDFPartitioningStage``::

        {"tar": "/data/arXiv_src_0001_001.tar",
         "member": "0001/astro-ph0001001.gz",
         "offset": 1536, "size": 29539}

    Parameters
    ----------
    papers_per_task
        Submissions per output task.  Bounds the peak memory of the reader
        stage, since it decompresses one task's worth of projects at a time.
    max_papers_per_tar
        If set, index at most this many submissions per tar (useful for smoke
        tests against a full shard).
    submission_extensions
        Outer-tar member suffixes treated as LaTeX submissions.
    read_kwargs
        Passed to :mod:`fsspec`; ``storage_options`` is honoured for remote tars.
    """

    papers_per_task: int = 100
    max_papers_per_tar: int | None = None
    submission_extensions: tuple[str, ...] = DEFAULT_SUBMISSION_EXTENSIONS
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "arxiv_tar_partitioning"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def __post_init__(self) -> None:
        if self.papers_per_task < 1:
            msg = f"papers_per_task must be >= 1, got {self.papers_per_task}"
            raise ValueError(msg)
        self._storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        from nemo_curator.backends.utils import RayStageSpecKeys

        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _index_tar(self, tar_path: str) -> list[str]:
        """Return one JSON entry per submission member of *tar_path*."""
        entries: list[str] = []
        skipped_pdf = 0
        with (
            fsspec.open(tar_path, mode="rb", **self._storage_options) as fobj,
            tarfile.open(fileobj=fobj, mode="r:*") as tf,
        ):
            for member in tf:
                if not member.isfile():
                    continue
                if not member.name.endswith(self.submission_extensions):
                    if member.name.lower().endswith(".pdf"):
                        skipped_pdf += 1
                    continue
                entries.append(
                    json.dumps(
                        {
                            "tar": tar_path,
                            "member": member.name,
                            "offset": member.offset_data,
                            "size": member.size,
                        }
                    )
                )
                if self.max_papers_per_tar is not None and len(entries) >= self.max_papers_per_tar:
                    break
        if skipped_pdf:
            logger.info("{}: skipped {} PDF-only submission(s) with no LaTeX source", tar_path, skipped_pdf)
        return entries

    def process(self, task: FileGroupTask) -> list[FileGroupTask]:
        tasks: list[FileGroupTask] = []
        for tar_path in task.data:
            try:
                entries = self._index_tar(tar_path)
            except (tarfile.TarError, OSError) as exc:
                # A corrupt shard must not take down the run; the rest still process.
                logger.warning("Failed to index {}: {}", tar_path, exc)
                continue

            emitted = 0
            for start in range(0, len(entries), self.papers_per_task):
                group = entries[start : start + self.papers_per_task]
                emitted += 1
                metadata = dict(task._metadata)
                metadata["source_files"] = [f"{tar_path}::papers_{start:06d}_{start + len(group):06d}"]
                metadata["partition_index"] = start // self.papers_per_task
                if self._storage_options:
                    metadata["source_storage_options"] = self._storage_options
                tasks.append(
                    FileGroupTask(
                        dataset_name=task.dataset_name,
                        data=group,
                        _metadata=metadata,
                        _stage_perf=task._stage_perf,
                    )
                )
            logger.info("{}: indexed {} submissions into {} task(s)", tar_path, len(entries), emitted)
        return tasks
