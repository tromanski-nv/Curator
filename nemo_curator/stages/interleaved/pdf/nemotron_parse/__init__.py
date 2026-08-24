# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""PDF to interleaved records, in two phases.

**Parse** -- :class:`NemotronParsePDFReader` renders each page, runs the model,
and decodes its output into one row per element: the text, its layout class,
its page and its bounding box.  That is the Nemotron-Parse element format, and
it is the artifact worth keeping, because reproducing it costs a GPU.

**Post-process** -- :class:`NemotronParseMarkdownPostprocessor` turns those
elements into a document: paragraphs rejoined across columns and pages, figures
next to their captions, page furniture and bibliographies marked, text written
as markdown.  Cheap, CPU-only, and the half you actually iterate on.

Which release you are running is a
:class:`~.versions.NemotronParseProfile`, resolved once and carried in task
metadata, so moving from v1.2 to a later release is one argument rather than a
hunt for version strings.
"""

from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import NemotronParsePDFReader
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import NemotronParseInferenceStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import PDFPartitioningStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocess import NemotronParsePostprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import (
    Config,
    NemotronParseMarkdownPostprocessor,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.preprocess import PDFPreprocessStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.versions import (
    DEFAULT_PROFILE,
    NemotronParseProfile,
    known_versions,
    register_profile,
)

__all__ = [
    "DEFAULT_PROFILE",
    "Config",
    "NemotronParseInferenceStage",
    "NemotronParseMarkdownPostprocessor",
    "NemotronParsePDFReader",
    "NemotronParsePostprocessStage",
    "NemotronParseProfile",
    "PDFPartitioningStage",
    "PDFPreprocessStage",
    "known_versions",
    "register_profile",
]
