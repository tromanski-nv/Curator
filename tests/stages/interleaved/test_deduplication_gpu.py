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

from pathlib import Path

import pandas as pd
import pytest

cudf = pytest.importorskip("cudf")

from nemo_curator.stages.deduplication.fuzzy.minhash import InterleavedMinHashStage  # noqa: E402
from nemo_curator.stages.interleaved.deduplication.exact_identification import (  # noqa: E402
    InterleavedExactDuplicateIdentification,
)

pytestmark = pytest.mark.gpu


def _interleaved_gpu_frame() -> "cudf.DataFrame":
    return cudf.DataFrame(
        {
            "sample_id": ["s_a", "s_a", "s_a", "s_b", "s_image"],
            "position": [2, 0, 1, 0, 0],
            "modality": ["text", "text", "image", "text", "image"],
            "text_content": ["world", "hello", None, "other", None],
        },
    )


def test_exact_and_fuzzy_reconstruct_identical_text(tmp_path: Path) -> None:
    df = _interleaved_gpu_frame()
    fuzzy_stage = InterleavedMinHashStage(
        output_path=str(tmp_path / "fuzzy"),
        text_mode="text_rows",
        text_separator="\n\n",
    )
    exact_stage = InterleavedExactDuplicateIdentification(
        output_path=str(tmp_path / "exact"),
        text_mode="text_rows",
        text_separator="\n\n",
    )

    fuzzy_text = fuzzy_stage._extract_text_rows(df.copy()).to_pandas().sort_values("sample_id").reset_index(drop=True)
    exact_text = exact_stage._extract_text_rows(df.copy()).to_pandas().sort_values("sample_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(fuzzy_text, exact_text)
    assert fuzzy_text.loc[fuzzy_text["sample_id"] == "s_a", "text"].iloc[0] == "hello\n\nworld"


def test_fuzzy_document_normalization_keeps_image_only_sample(tmp_path: Path) -> None:
    stage = InterleavedMinHashStage(
        output_path=str(tmp_path / "fuzzy"),
        text_mode="text_rows",
    )
    documents = stage._extract_documents(_interleaved_gpu_frame()).to_pandas().sort_values("sample_id")

    assert documents["sample_id"].tolist() == ["s_a", "s_b", "s_image"]
    assert pd.isna(documents.loc[documents["sample_id"] == "s_image", "text"].iloc[0])
