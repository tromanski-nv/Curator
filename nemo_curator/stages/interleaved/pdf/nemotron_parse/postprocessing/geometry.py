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

"""Bounding-box helpers and caption matching.

Nemotron-Parse carries normalised bboxes on every element, so the geometric
parts of the pipeline are exact rather than approximated by reading order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import BBox

#: Cost used for a pair whose bbox is missing, so the assignment solver ranks
#: it below every real pair rather than rejecting the whole problem.
_MISSING_BBOX_COST = 1e6


def merge(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def x_overlap(a: BBox, b: BBox) -> bool:
    """Do the two boxes share any horizontal extent?

    Used as a column test: two stacked boxes in the same column overlap in x,
    so the gap between them is an ordinary paragraph break.  Boxes that do NOT
    overlap in x sit in different columns, so the break is a layout artefact
    and the text may actually run on.
    """
    return a[0] <= b[2] and a[2] >= b[0]


def distance(a: BBox, b: BBox) -> float:
    """0 when the boxes intersect, else Manhattan distance between borders."""
    if a[2] >= b[0] and a[0] <= b[2] and a[3] >= b[1] and a[1] <= b[3]:
        return 0.0
    dx = min(abs(a[0] - b[2]), abs(a[2] - b[0]))
    dy = min(abs(a[1] - b[3]), abs(a[3] - b[1]))
    return dx + dy


def match_captions(
    figure_boxes: list[int],
    caption_boxes: list[int],
    bboxes: dict[int, BBox | None],
) -> dict[int, int]:
    """Assign each caption to a figure or table, minimising total bbox distance.

    Uses the Hungarian algorithm when SciPy is available.  Falls back to greedy
    nearest-pair otherwise, so this package has no hard scientific dependency.
    The two agree on the ordinary page and diverge in two places: when two
    captions contend for the same figure, and when a figure or caption has no
    bounding box at all -- the solver is handed a large finite cost for those
    and may still assign one, where greedy skips the pair outright.

    Returns a mapping of figure position -> caption position.
    """
    if not figure_boxes or not caption_boxes:
        return {}

    pairs = [
        (f, c) for f in figure_boxes for c in caption_boxes if bboxes.get(f) is not None and bboxes.get(c) is not None
    ]
    if not pairs:
        return {}

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _match_greedy(pairs, bboxes)

    cost = np.array(
        [
            [
                distance(bboxes[f], bboxes[c])
                if bboxes.get(f) is not None and bboxes.get(c) is not None
                else _MISSING_BBOX_COST
                for c in caption_boxes
            ]
            for f in figure_boxes
        ]
    )
    rows, cols = linear_sum_assignment(cost)
    return {figure_boxes[r]: caption_boxes[c] for r, c in zip(rows, cols, strict=False)}


def _match_greedy(
    pairs: list[tuple[int, int]],
    bboxes: dict[int, BBox | None],
) -> dict[int, int]:
    """Nearest-pair matching, used when SciPy is not installed."""
    assigned: dict[int, int] = {}
    used: set[int] = set()
    ranked = sorted(((f, c, distance(bboxes[f], bboxes[c])) for f, c in pairs), key=lambda t: t[2])
    for f, c, _ in ranked:
        if f in assigned or c in used:
            continue
        assigned[f] = c
        used.add(c)
    return assigned
