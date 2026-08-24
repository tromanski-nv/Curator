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

"""Post-processing for Nemotron-Parse output: elements in, a document out.

The model returns what it found on each page -- a box of text, its class, its
bounding box -- in roughly reading order.  That is not yet a document.  A
sentence is cut in half by a column break; a figure sits in the middle of a
paragraph; the page number is a box like any other; the bibliography reads as
prose.  This package applies the rules that turn the one into the other, and
writes the result as markdown in the interleaved schema.

Two ways in.  The pipeline stage::

    from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import (
        Config, NemotronParseMarkdownPostprocessor,
    )

    pipeline.add_stage(NemotronParseMarkdownPostprocessor(config=Config()))

and the pure functions underneath it, which know nothing about Curator and can
be run on a list of dataclasses in a notebook::

    from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import (
        Element, postprocess, to_markdown,
    )

    doc = postprocess(elements, Config(min_caption_chars=80))
    doc.text          # what a training pipeline consumes: kept boxes only
    doc.boxes         # every box, including the discarded ones, and why
    to_markdown(doc.boxes)

Not to be confused with :mod:`..postprocess`, the module next door: that one
decodes the model's raw output string into the element rows this package takes
as *input*.  It runs first, and it belongs to the parse phase.

Ported from the post-processing that accompanied the PDF-extraction model
Nemotron-Parse replaced.  Two of its stages are deliberately absent --
Tesseract F1 scoring needs rendered pages, and document splitting was
triggered by page-F1 failure and by runs of unparseable ``Bad-box`` output,
neither of which a model that emits structured elements produces.
"""

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.composite import (
    NemotronParseMarkdownPostprocessor,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.markdown import (
    RENDERERS,
    render,
    to_markdown,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import (
    FLOAT_CLASSES,
    FLUSH_CLASSES,
    PAGE_FURNITURE,
    BBox,
    Box,
    Config,
    Element,
    Layout,
    Page,
    ProcessedDocument,
    Stats,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.stages import (
    ElementCleaningStage,
    FloatAssignmentStage,
    FusedPostprocessingStage,
    MarkdownAssemblyStage,
    PageFurnitureStage,
    ParagraphReconstitutionStage,
    SectionSkippingStage,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.steps import (
    assemble,
    assign_floats,
    clean,
    mark_page_furniture,
    postprocess,
    reconstitute_paragraphs,
    skip_sections,
    summarise,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.text import (
    count_words,
    is_to_be_continued,
    strip_markdown,
)

__all__ = [
    "FLOAT_CLASSES",
    "FLUSH_CLASSES",
    "PAGE_FURNITURE",
    "RENDERERS",
    "BBox",
    "Box",
    "Config",
    "Element",
    "ElementCleaningStage",
    "FloatAssignmentStage",
    "FusedPostprocessingStage",
    "Layout",
    "MarkdownAssemblyStage",
    "NemotronParseMarkdownPostprocessor",
    "Page",
    "PageFurnitureStage",
    "ParagraphReconstitutionStage",
    "ProcessedDocument",
    "SectionSkippingStage",
    "Stats",
    "assemble",
    "assign_floats",
    "clean",
    "count_words",
    "is_to_be_continued",
    "mark_page_furniture",
    "postprocess",
    "reconstitute_paragraphs",
    "render",
    "skip_sections",
    "strip_markdown",
    "summarise",
    "to_markdown",
]
