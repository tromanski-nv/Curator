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

"""Composite reader for raw arXiv LaTeX source shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.latex.arxiv.regex.partitioning import (
    DEFAULT_SUBMISSION_EXTENSIONS,
    ArxivTarPartitioningStage,
)
from nemo_curator.stages.interleaved.latex.arxiv.regex.reader import (
    DEFAULT_MAX_PROJECT_BYTES,
    ArxivLatexReaderStage,
)
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.tasks import EmptyTask, InterleavedBatch

#: arXiv publishes bulk source as ``arXiv_src_YYMM_NNN.tar``.
DEFAULT_ARXIV_TAR_EXTENSIONS: tuple[str, ...] = (".tar",)


@dataclass
class ArxivLatexReader(CompositeStage[EmptyTask, InterleavedBatch]):
    r"""Extract interleaved text and images from raw arXiv LaTeX source tars.

    arXiv's bulk source dumps (``s3://arxiv/src/arXiv_src_YYMM_NNN.tar``) hold
    one gzipped LaTeX project per submission.  This reader walks those shards,
    parses each project's LaTeX, and emits an :class:`InterleavedBatch` whose
    rows follow the document's reading order -- body text, figure, caption,
    body text, ... -- ready for the existing interleaved filters and writers.

    Decomposes into three stages:

    1. :class:`FilePartitioningStage` -- group shard paths into tasks
    2. :class:`ArxivTarPartitioningStage` -- index each shard, slice it into
       fixed-size groups of submissions
    3. :class:`ArxivLatexReaderStage` -- decompress, parse, and emit rows

    Example::

        pipeline = Pipeline(name="arxiv_latex")
        pipeline.add_stage(ArxivLatexReader(file_paths="/data/arxiv-src/"))
        pipeline.add_stage(InterleavedParquetWriterStage(path="/out/", mode="overwrite"))
        pipeline.run()

    Figures are read as stored.  Pre-2005 submissions are dominated by
    PostScript, so pass ``image_content_types=("image/png", "image/jpeg")`` to
    keep only figures that are already raster, or handle conversion downstream.

    Parameters
    ----------
    file_paths
        Directory or list of ``arXiv_src_*.tar`` shards.  Local or remote.
    files_per_partition, blocksize
        Shard-level grouping, as on any Curator reader.  Shards are ~500 MB, so
        the default of one per task is usually right.
    papers_per_task
        Submissions handled per reader task.  Controls peak memory: a task
        decompresses this many projects and holds their figure bytes.
    max_papers_per_tar
        Cap submissions indexed per shard (handy for smoke tests).
    read_kwargs
        Passed to :mod:`fsspec`; ``storage_options`` is honoured.
    dataset_name
        Name assigned to output tasks.

    The remaining parameters are forwarded to :class:`ArxivLatexReaderStage`;
    see that class for their meaning.
    """

    file_paths: str | list[str]
    files_per_partition: int | None = 1
    blocksize: int | str | None = None
    file_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_ARXIV_TAR_EXTENSIONS))
    submission_extensions: tuple[str, ...] = DEFAULT_SUBMISSION_EXTENSIONS
    papers_per_task: int = 100
    max_papers_per_tar: int | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)

    include_images: bool = True
    emit_captions: bool = True
    clean: bool = True
    drop_bibliography: bool = True
    drop_appendix: bool = False
    min_text_chars: int = 1
    max_image_bytes: int | None = None
    image_content_types: tuple[str, ...] | None = None
    max_project_bytes: int = DEFAULT_MAX_PROJECT_BYTES
    on_missing_graphics: Literal["skip", "annotate"] = "skip"
    max_batch_bytes: int | None = None

    _generate_ids: bool = False
    _assign_ids: bool = False
    name: str = "arxiv_latex_reader"

    def __post_init__(self) -> None:
        super().__init__()
        self.storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.files_per_partition,
                blocksize=self.blocksize,
                file_extensions=self.file_extensions,
                storage_options=self.storage_options,
            ),
            ArxivTarPartitioningStage(
                papers_per_task=self.papers_per_task,
                max_papers_per_tar=self.max_papers_per_tar,
                submission_extensions=self.submission_extensions,
                read_kwargs=self.read_kwargs,
            ),
            ArxivLatexReaderStage(
                read_kwargs=self.read_kwargs,
                include_images=self.include_images,
                emit_captions=self.emit_captions,
                clean=self.clean,
                drop_bibliography=self.drop_bibliography,
                drop_appendix=self.drop_appendix,
                min_text_chars=self.min_text_chars,
                max_image_bytes=self.max_image_bytes,
                image_content_types=self.image_content_types,
                max_project_bytes=self.max_project_bytes,
                on_missing_graphics=self.on_missing_graphics,
                max_batch_bytes=self.max_batch_bytes,
                _generate_ids=self._generate_ids,
                _assign_ids=self._assign_ids,
            ),
        ]
