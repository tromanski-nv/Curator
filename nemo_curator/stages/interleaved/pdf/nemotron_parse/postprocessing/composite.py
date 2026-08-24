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

"""The post-processing phase, as one stage."""

from __future__ import annotations

from dataclasses import dataclass, field

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import Config
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.stages import (
    ElementCleaningStage,
    FloatAssignmentStage,
    FusedPostprocessingStage,
    MarkdownAssemblyStage,
    PageFurnitureStage,
    ParagraphReconstitutionStage,
    SectionSkippingStage,
)
from nemo_curator.tasks import InterleavedBatch


@dataclass
class NemotronParseMarkdownPostprocessor(CompositeStage[InterleavedBatch, InterleavedBatch]):
    """Turn Nemotron-Parse elements into an interleaved markdown document.

    The second half of the PDF pipeline.  Its input is what
    :class:`~nemo_curator.stages.interleaved.pdf.nemotron_parse.NemotronParsePDFReader`
    emits -- one row per element the model found, in the model's own reading
    order, with its class, page and bounding box.  Its output is the same
    interleaved schema, but as a document: paragraphs whole, figures next to
    their captions, running heads and bibliographies marked, text written as
    markdown, and pictures still in place in the reading order.

    Decomposes into six execution stages, one per rule it applies:

    1. :class:`~.stages.ElementCleaningStage` -- condemn what never had content
    2. :class:`~.stages.FloatAssignmentStage` -- floats out of the flow,
       captions matched to figures
    3. :class:`~.stages.PageFurnitureStage` -- running heads and page numbers
    4. :class:`~.stages.SectionSkippingStage` -- contents and bibliography
    5. :class:`~.stages.ParagraphReconstitutionStage` -- rejoin broken sentences
    6. :class:`~.stages.MarkdownAssemblyStage` -- flow before floats, as markdown

    Example -- the whole pipeline, PDFs to markdown::

        pipeline = Pipeline(name="nemotron_parse_markdown")
        pipeline.add_stage(NemotronParsePDFReader(manifest_path=..., pdf_dir=...))
        pipeline.add_stage(NemotronParseMarkdownPostprocessor())
        pipeline.add_stage(InterleavedParquetWriterStage(path="/out/"))

    Example -- post-processing alone, over parse output already on disk.  This
    is the reason the two phases are separate stages: the parse phase costs a
    GPU and the rules are what you actually want to iterate on::

        pipeline.add_stage(InterleavedParquetReader(file_paths="/parsed/"))
        pipeline.add_stage(NemotronParseMarkdownPostprocessor(
            config=Config(skip_toc_bib=False),
        ))

    A document must arrive whole: every row of a ``sample_id`` in one batch.
    Paragraph reconstitution reaches across pages, so a document split between
    two batches would come out split.  The parse phase emits a document's rows
    together and the Parquet writer keeps a task's rows in one file, so reading
    that output back with ``files_per_partition=1`` preserves it.

    Parameters
    ----------
    config
        The tunables -- see :class:`~.model.Config`.  Every rule can be
        switched off independently, which is what makes a before/after
        comparison mean something.  Passed as an object rather than flattened
        onto this stage because the rules and their defaults are what was
        ported, and they are documented where they are defined.
    emit_dropped
        Keep condemned boxes in the output, marked with ``keep=False`` and
        ``drop_reason``.  Off for a training corpus; on for a viewer that has
        to show what was thrown away and why.
    fuse
        Run all six steps in one stage instead of six.  Faster -- no
        serialisation between steps -- but the intermediate state is no longer
        observable, so a run cannot be stopped after step 3 to see what it did.
    """

    config: Config = field(default_factory=Config)
    emit_dropped: bool = False
    fuse: bool = False
    name: str = "nemotron_parse_markdown_postprocessor"

    def __post_init__(self) -> None:
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        if self.fuse:
            return [FusedPostprocessingStage(config=self.config, emit_dropped=self.emit_dropped)]
        return [
            ElementCleaningStage(config=self.config),
            FloatAssignmentStage(config=self.config),
            PageFurnitureStage(config=self.config),
            SectionSkippingStage(config=self.config),
            ParagraphReconstitutionStage(config=self.config),
            MarkdownAssemblyStage(config=self.config, emit_dropped=self.emit_dropped),
        ]

    def get_description(self) -> str:
        shape = "fused" if self.fuse else "six steps"
        return f"Nemotron-Parse elements -> interleaved markdown ({shape})"
