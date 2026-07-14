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

from nemo_curator.stages.interleaved.deduplication.pdf_sha import PdfSha256InventoryStage
from nemo_curator.stages.interleaved.deduplication.removal import (
    InterleavedSampleDuplicatesRemovalStage,
    InterleavedSampleIdRemovalStage,
)
from nemo_curator.stages.interleaved.deduplication.removal_workflow import InterleavedDuplicatesRemovalWorkflow

__all__ = [
    "InterleavedDuplicatesRemovalWorkflow",
    "InterleavedExactDeduplicationWorkflow",
    "InterleavedSampleDuplicatesRemovalStage",
    "InterleavedSampleIdRemovalStage",
    "InterleavedTextFuzzyDeduplicationWorkflow",
    "PdfSha256InventoryStage",
]


def __getattr__(name: str) -> type:
    """Load the GPU fuzzy workflow only when requested."""
    if name == "InterleavedTextFuzzyDeduplicationWorkflow":
        from nemo_curator.stages.interleaved.deduplication.fuzzy_workflow import (
            InterleavedTextFuzzyDeduplicationWorkflow,
        )

        return InterleavedTextFuzzyDeduplicationWorkflow
    if name == "InterleavedExactDeduplicationWorkflow":
        from nemo_curator.stages.interleaved.deduplication.exact_workflow import (
            InterleavedExactDeduplicationWorkflow,
        )

        return InterleavedExactDeduplicationWorkflow
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
