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

"""CPU postprocess stage: parse model output, align images, build interleaved rows.

This is the last stage of the *parse* phase -- it decodes the model's raw
output string into one row per element.  The rules that turn those elements
into a document live next door in
:mod:`~nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing`,
which reads what this module writes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa
from PIL import Image

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import versions
from nemo_curator.stages.interleaved.pdf.nemotron_parse.utils import (
    DEFAULT_MIN_CROP_PX,
    build_interleaved_rows,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA


@dataclass
class NemotronParsePostprocessStage(ProcessingStage[InterleavedBatch, InterleavedBatch]):
    """CPU stage: parse raw model output and build the final interleaved schema.

    Reads page images from ``binary_content`` and raw Nemotron-Parse output
    from ``text_content``, then constructs one row per element (text, image,
    table, metadata) in the interleaved schema.

    Floater reordering (Pictures/Captions) is applied for releases that emit
    them at the end of the page rather than in reading order -- v1.1 does,
    v1.2 and later do not.  Which is which is read off the release profile,
    not sniffed out of the model path here.

    Parameters
    ----------
    profile
        The Nemotron-Parse release being run.  Set by the composite, which
        resolves it once in the driver.  Left ``None`` the profile is recovered
        from task metadata instead -- correct for the releases this repo knows
        about, but a profile added at runtime by
        :func:`~.versions.register_profile` exists only in the process that
        registered it, so a stage constructed without one would not see it.
    proc_size
        Default model processor size ``(height, width)``.  Overridden at
        runtime by ``task._metadata["proc_size"]`` when available.
    min_crop_px
        Minimum pixel dimension for image crops.  Smaller crops (typically
        degenerate bboxes) are filtered out.
    """

    proc_size: tuple[int, int] = (2048, 1664)
    min_crop_px: int = DEFAULT_MIN_CROP_PX
    profile: versions.NemotronParseProfile | None = None
    name: str = "nemotron_parse_postprocess"
    resources: Resources = field(default_factory=lambda: Resources(cpus=2.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def process(self, task: InterleavedBatch) -> InterleavedBatch | None:
        pages = task.to_pandas()
        proc_size = tuple(task._metadata.get("proc_size", self.proc_size))
        profile = self.profile or versions.resolve(
            task._metadata.get("parse_version"), task._metadata.get("model_path")
        )
        reorder = not profile.floats_in_reading_order

        all_rows: list[dict[str, Any]] = []
        for sample_id, sample_group in pages.groupby("sample_id", sort=False):
            sorted_group = sample_group.sort_values("position")
            url = str(sorted_group["url"].iloc[0])
            pdf_name = str(sorted_group["pdf_name"].iloc[0])

            page_images = [Image.open(io.BytesIO(b)) for b in sorted_group["binary_content"]]
            page_outputs = [str(t) if t else "" for t in sorted_group["text_content"].tolist()]

            all_rows.extend(
                build_interleaved_rows(
                    str(sample_id),
                    url,
                    pdf_name,
                    page_images,
                    page_outputs,
                    proc_size,
                    reorder_floaters=reorder,
                    min_crop_px=self.min_crop_px,
                )
            )

        if not all_rows:
            return None

        final_df = pd.DataFrame(all_rows)
        for col in INTERLEAVED_SCHEMA.names:
            if col not in final_df.columns:
                final_df[col] = None

        return InterleavedBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pandas(final_df, preserve_index=False),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )
