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

"""The domain types: defaults, derived properties, and the values written to Parquet."""

import pytest

from nemo_curator.stages.interleaved.latex.arxiv.latexml.model import (
    ARTIFACT_RATE_WARN,
    ArtifactReport,
    Assessment,
    ConvertedDocument,
    HtmlCounts,
    Status,
    Tier,
)


# --- enum values are a storage format ---


def test_tier_values_are_stable():
    """These strings are written into Parquet; changing one silently re-labels a corpus."""
    assert [t.value for t in Tier] == ["A", "B", "C", "rejected"]


def test_status_values_are_lowercase_identifiers():
    for status in Status:
        assert status.value == status.value.lower()
        assert " " not in status.value


def test_timeout_is_its_own_status():
    """A timeout says nothing about whether the document *could* convert."""
    assert Status.TIMEOUT is not Status.FATAL


# --- Assessment ---


@pytest.mark.parametrize(
    ("tier", "usable"),
    [
        pytest.param(Tier.A, True, id="A"),
        pytest.param(Tier.B, True, id="B"),
        pytest.param(Tier.C, True, id="C"),
        pytest.param(Tier.REJECTED, False, id="rejected"),
    ],
)
def test_usable_is_everything_but_rejected(tier, usable):
    assert Assessment(status=Status.OK, tier=tier, counts=HtmlCounts()).usable is usable


@pytest.mark.parametrize(
    ("tier", "gates", "n_error", "expected"),
    [
        pytest.param(Tier.C, (), 3, "errors", id="errors_only"),
        pytest.param(Tier.C, ("residual_latex",), 0, "artifacts", id="artifacts_only"),
        pytest.param(Tier.C, ("residual_latex",), 2, "errors+artifacts", id="both"),
        pytest.param(Tier.B, ("residual_latex",), 2, None, id="not_tier_c"),
    ],
)
def test_tier_c_reason_separates_causes(tier, gates, n_error, expected):
    """status alone cannot answer this: both causes report suspect_artifacts."""
    assessment = Assessment(status=Status.OK, tier=tier, counts=HtmlCounts(), failed_gates=gates)
    assert assessment.tier_c_reason(n_error) == expected


# --- ArtifactReport ---


def test_rate_is_zero_for_an_empty_document():
    assert ArtifactReport().rate == 0.0


def test_rate_is_artifacts_per_character():
    assert ArtifactReport(text_chars=1000, total=5).rate == pytest.approx(0.005)


def test_structural_artifacts_count_regardless_of_rate():
    """A leaked \\ref in an author list is a defect at any document size."""
    report = ArtifactReport(text_chars=10**6, total=1, by_kind={"ref": 1})
    assert report.structural == 1
    assert report.rate < ARTIFACT_RATE_WARN
    assert report.notable


def test_spacing_noise_is_notable_only_in_volume():
    sparse = ArtifactReport(text_chars=10**6, total=1, by_kind={"hbox": 1})
    dense = ArtifactReport(text_chars=1000, total=50, by_kind={"hbox": 50})
    assert not sparse.notable
    assert dense.notable


# --- ConvertedDocument ---


def test_defaults_describe_a_submission_that_produced_nothing():
    doc = ConvertedDocument(arxiv_id="1234.5678", source_sha256="a" * 64)
    assert not doc.converted
    assert doc.status is Status.NO_SOURCE
    assert doc.tier is Tier.REJECTED
    assert doc.source_expects_math is None
    assert doc.failed_gates == ()
    assert doc.n_assets == 0


def test_converted_reflects_html_presence_not_usability():
    doc = ConvertedDocument(arxiv_id="x", source_sha256="y", html="<html></html>", tier=Tier.REJECTED)
    assert doc.converted
