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

"""Tests for the post-processing stages and the composite that runs them.

Stages are driven by calling ``process()`` directly, so no Ray cluster is
involved.  The rules themselves are tested in ``test_steps.py``; what is tested
here is the plumbing -- that a document survives the round trip through Arrow
between every pair of stages, that the six stages agree with the fused one, and
that what comes out is the interleaved schema.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import rows as rows_module
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.composite import (
    NemotronParseMarkdownPostprocessor,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import Config
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.stages import (
    STATS_KEY,
    ElementCleaningStage,
    FloatAssignmentStage,
    FusedPostprocessingStage,
    MarkdownAssemblyStage,
    PageFurnitureStage,
    ParagraphReconstitutionStage,
    SectionSkippingStage,
)
from nemo_curator.tasks import InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

if TYPE_CHECKING:
    from nemo_curator.stages.base import ProcessingStage

SAMPLE = "2410.10730"
PNG_BYTES = b"\x89PNG\r\n\x1a\nnot-a-real-image"

LEFT_COLUMN = [0.08, 0.20, 0.45, 0.40]
RIGHT_COLUMN = [0.55, 0.20, 0.92, 0.40]
FULL_WIDTH = [0.08, 0.20, 0.92, 0.40]

#: A caption the rules will keep: over 100 characters and over 10 spaces.
LONG_CAPTION = (
    "FIG. 1. The emitter sits a distance d above the dielectric slab, whose permittivity "
    "varies with frequency across the range considered in this work."
)


def _metadata_row(sample_id: str = SAMPLE, num_pages: int = 2) -> dict[str, Any]:
    """The per-document row the parse phase emits at position -1."""
    return {
        "sample_id": sample_id,
        "position": -1,
        "modality": "metadata",
        "content_type": "application/json",
        "text_content": json.dumps(
            {"url": f"http://x/{sample_id}", "pdf_name": f"{sample_id}.pdf", "num_pages": num_pages}
        ),
        "binary_content": None,
        "source_ref": None,
        "materialize_error": None,
        "url": f"http://x/{sample_id}",
        "page_number": None,
        "pdf_name": f"{sample_id}.pdf",
        "element_class": None,
    }


def _element_row(  # noqa: PLR0913 -- a row factory mirrors the columns of a row
    position: int,
    element_class: str,
    text: str,
    page: int = 0,
    bbox: list[float] | None = None,
    *,
    sample_id: str = SAMPLE,
    binary: bytes | None = None,
) -> dict[str, Any]:
    """One row of the Nemotron-Parse element format."""
    if element_class == "Picture":
        modality, content_type = "image", "image/png"
    elif element_class == "Table":
        modality, content_type = "table", "text/markdown"
    else:
        modality, content_type = "text", "text/markdown"
    return {
        "sample_id": sample_id,
        "position": position,
        "modality": modality,
        "content_type": content_type,
        "text_content": text,
        "binary_content": binary,
        "source_ref": json.dumps({"page": page, "bbox": bbox if bbox is not None else FULL_WIDTH}),
        "materialize_error": None,
        "url": f"http://x/{sample_id}",
        "page_number": page,
        "pdf_name": f"{sample_id}.pdf",
        "element_class": element_class,
    }


def _batch(records: list[dict[str, Any]]) -> InterleavedBatch:
    return InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist(records))


def _paper() -> list[dict[str, Any]]:
    """A two-page paper exercising every rule at least once."""
    return [
        _metadata_row(),
        _element_row(0, "Title", "Casimir Forces on a Quantum Emitter", page=0),
        _element_row(1, "Section-header", "I. INTRODUCTION", page=0),
        _element_row(2, "Text", "We propose a model for a quantum emitter interacting", page=0, bbox=LEFT_COLUMN),
        _element_row(3, "Text", r"with a dispersive object, via \(\hat{H}_a\).", page=0, bbox=RIGHT_COLUMN),
        _element_row(4, "Picture", "", page=0, bbox=[0.10, 0.55, 0.45, 0.80], binary=PNG_BYTES),
        _element_row(5, "Caption", LONG_CAPTION, page=0, bbox=[0.10, 0.81, 0.45, 0.86]),
        _element_row(6, "Page-footer", "2", page=1, bbox=[0.48, 0.95, 0.52, 0.97]),
        _element_row(7, "List-item", "• the first consequence.", page=1),
        _element_row(8, "Section-header", "REFERENCES", page=1),
        _element_row(9, "Text", "[1] A. Someone, Phys. Rev. A 12, 345 (2020).", page=1),
    ]


def _run(stages: list[ProcessingStage], task: InterleavedBatch) -> InterleavedBatch | None:
    for stage in stages:
        task = stage.process(task)
        if task is None:
            return None
    return task


def _rows(task: InterleavedBatch | None) -> list[dict[str, Any]]:
    return [] if task is None else task.to_pyarrow().to_pylist()


def _classes(task: InterleavedBatch | None) -> list[str | None]:
    return [r["element_class"] for r in _rows(task) if r["position"] >= 0]


ALL_STAGES = [
    ElementCleaningStage,
    FloatAssignmentStage,
    PageFurnitureStage,
    SectionSkippingStage,
    ParagraphReconstitutionStage,
    MarkdownAssemblyStage,
    FusedPostprocessingStage,
]


# ---- the stage contract -----------------------------------------------------


class TestStageContract:
    @pytest.mark.parametrize("stage_class", ALL_STAGES)
    def test_every_stage_reads_and_writes_data(self, stage_class: type[ProcessingStage]) -> None:
        stage = stage_class()

        assert stage.inputs() == (["data"], [])
        assert stage.outputs() == (["data"], [])

    @pytest.mark.parametrize("stage_class", ALL_STAGES)
    def test_every_stage_has_its_own_name(self, stage_class: type[ProcessingStage]) -> None:
        assert stage_class().name.startswith("nemotron_parse_")

    def test_the_stage_names_are_unique(self) -> None:
        """``CompositeStage.with_()`` addresses child stages by name, and
        refuses to run at all if two of them share one."""
        names = [stage.name for stage in NemotronParseMarkdownPostprocessor().decompose()]

        assert len(set(names)) == len(names)

    @pytest.mark.parametrize("stage_class", ALL_STAGES)
    def test_a_batch_with_no_rows_produces_no_task(self, stage_class: type[ProcessingStage]) -> None:
        empty = InterleavedBatch(dataset_name="test", data=pa.Table.from_pylist([], schema=INTERLEAVED_SCHEMA))

        assert stage_class().process(empty) is None


# ---- the composite ----------------------------------------------------------


class TestComposite:
    def test_it_decomposes_into_one_stage_per_rule(self) -> None:
        composite = NemotronParseMarkdownPostprocessor()

        assert [type(s) for s in composite.decompose()] == [
            ElementCleaningStage,
            FloatAssignmentStage,
            PageFurnitureStage,
            SectionSkippingStage,
            ParagraphReconstitutionStage,
            MarkdownAssemblyStage,
        ]

    def test_fusing_collapses_it_to_one_stage(self) -> None:
        assert [type(s) for s in NemotronParseMarkdownPostprocessor(fuse=True).decompose()] == [
            FusedPostprocessingStage
        ]

    def test_the_config_reaches_every_child(self) -> None:
        config = Config(min_caption_chars=7, skip_toc_bib=False)

        assert all(stage.config is config for stage in NemotronParseMarkdownPostprocessor(config=config).decompose())

    def test_emit_dropped_reaches_the_stage_that_needs_it(self) -> None:
        stages = NemotronParseMarkdownPostprocessor(emit_dropped=True).decompose()

        assert stages[-1].emit_dropped is True

    def test_its_inputs_and_outputs_come_from_its_children(self) -> None:
        composite = NemotronParseMarkdownPostprocessor()

        assert composite.inputs() == (["data"], [])
        assert composite.outputs() == (["data"], [])


# ---- what comes out ---------------------------------------------------------


class TestOutput:
    def test_the_document_reads_in_order_with_its_figure_after_its_page(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        assert _classes(result) == [
            "Title",
            "Section-header",
            "Text",
            "Picture",
            "Caption",
            "List-item",
        ]

    def test_positions_are_renumbered_from_zero_with_the_metadata_row_first(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        assert [r["position"] for r in _rows(result)] == [-1, 0, 1, 2, 3, 4, 5]

    def test_a_broken_sentence_is_rejoined_and_says_where_from(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        joined = next(r for r in _rows(result) if r["element_class"] == "Text")
        assert joined["text_content"] == (
            r"We propose a model for a quantum emitter interacting with a dispersive object, via $\hat{H}_a$."
        )
        assert json.loads(joined["source_positions"]) == [2, 3]

    def test_text_rows_are_markdown_and_headings_carry_their_hashes(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))
        by_class = {r["element_class"]: r for r in _rows(result) if r["position"] >= 0}

        assert by_class["Title"]["text_content"] == "# Casimir Forces on a Quantum Emitter"
        assert by_class["Section-header"]["text_content"] == "## I. INTRODUCTION"
        assert by_class["List-item"]["text_content"] == "- the first consequence."
        assert by_class["Title"]["content_type"] == "text/markdown"

    def test_a_picture_keeps_its_bytes_and_its_own_content_type(self) -> None:
        """An interleaved document is text *and* pictures in one reading order."""
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        picture = next(r for r in _rows(result) if r["element_class"] == "Picture")
        assert picture["modality"] == "image"
        assert picture["content_type"] == "image/png"
        assert picture["binary_content"] == PNG_BYTES

    def test_a_caption_points_at_the_figure_row_it_was_matched_to(self) -> None:
        """``matched_to`` has to be usable as a join key against ``position``,
        which means the emitted numbering, not the element numbering it was
        recorded in."""
        rows_out = _rows(_run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper())))

        caption = next(r for r in rows_out if r["element_class"] == "Caption")
        picture = next(r for r in rows_out if r["element_class"] == "Picture")
        assert caption["matched_to"] == picture["position"]

    def test_the_page_footer_and_the_bibliography_are_gone(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        assert "Page-footer" not in _classes(result)
        assert not any("Phys. Rev." in (r["text_content"] or "") for r in _rows(result))

    def test_the_metadata_row_records_what_the_rules_did(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        metadata = json.loads(_rows(result)[0]["text_content"])
        assert metadata["pdf_name"] == f"{SAMPLE}.pdf"  # the parse phase's own fields survive
        assert metadata[STATS_KEY]["paragraphs_reconstituted"] == 1
        assert metadata[STATS_KEY]["captions_matched"] == 1
        assert metadata[STATS_KEY]["boxes_in"] == 10

    def test_passthrough_columns_survive_a_step_that_rewrote_the_text(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        assert {r["pdf_name"] for r in _rows(result)} == {f"{SAMPLE}.pdf"}
        assert {r["url"] for r in _rows(result)} == {f"http://x/{SAMPLE}"}

    def test_the_output_is_the_interleaved_schema_with_named_extras(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        assert result.to_pyarrow().schema.names == [
            *INTERLEAVED_SCHEMA.names,
            # the columns this package writes, in EXTRA_SCHEMA order...
            "element_class",
            "page_number",
            "source_positions",
            "matched_to",
            # ...then whatever the parse phase brought along
            "url",
            "pdf_name",
        ]
        assert result.validate() is True

    def test_the_working_columns_do_not_reach_the_output(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))

        assert rows_module.SLOT not in result.to_pyarrow().schema.names
        assert rows_module.KEEP not in result.to_pyarrow().schema.names


class TestEmitDropped:
    def test_the_condemned_boxes_come_back_with_the_reason(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor(emit_dropped=True).decompose(), _batch(_paper()))

        verdicts = {r["element_class"]: (r["keep"], r["drop_reason"]) for r in _rows(result) if r["position"] >= 0}
        assert verdicts["Page-footer"] == (False, "page-furniture")
        assert verdicts["Title"] == (True, "")

    def test_nothing_the_model_found_is_missing(self) -> None:
        """The promise of the whole package: marked, never deleted."""
        result = _run(NemotronParseMarkdownPostprocessor(emit_dropped=True).decompose(), _batch(_paper()))

        covered = {p for r in _rows(result) if r["position"] >= 0 for p in json.loads(r["source_positions"])}
        assert covered == set(range(10))


# ---- the rules must not lose a picture -------------------------------------


class TestImagesSurvive:
    """An image carries its payload in a column the rules never see.  Anything
    that judges it by its text alone deletes a picture."""

    @staticmethod
    def _one_figure_mid_paragraph() -> list[dict[str, Any]]:
        return [
            _metadata_row(),
            _element_row(0, "Text", "a sentence that runs on", page=0, bbox=LEFT_COLUMN),
            _element_row(1, "Picture", "", page=0, bbox=[0.4, 0.4, 0.6, 0.6], binary=PNG_BYTES),
            _element_row(2, "Text", "and finishes.", page=0, bbox=RIGHT_COLUMN),
        ]

    @pytest.mark.parametrize("assign_floats", [True, False])
    def test_a_picture_in_the_reading_flow_keeps_its_bytes(self, assign_floats: bool) -> None:
        """With ``assign_floats=False`` the picture is never lifted out of the
        flow, so paragraph reconstitution meets it head on."""
        config = Config(assign_floats=assign_floats)

        rows_out = _rows(
            _run(
                NemotronParseMarkdownPostprocessor(config=config).decompose(), _batch(self._one_figure_mid_paragraph())
            )
        )

        images = [r for r in rows_out if r["binary_content"]]
        assert len(images) == 1
        assert images[0]["binary_content"] == PNG_BYTES

    @pytest.mark.parametrize("assign_floats", [True, False])
    def test_prose_still_runs_on_across_it(self, assign_floats: bool) -> None:
        config = Config(assign_floats=assign_floats)

        rows_out = _rows(
            _run(
                NemotronParseMarkdownPostprocessor(config=config).decompose(), _batch(self._one_figure_mid_paragraph())
            )
        )

        assert [r["text_content"] for r in rows_out if r["element_class"] == "Text"] == [
            "a sentence that runs on and finishes."
        ]


# ---- the output schema must not depend on what the batch happened to hold ---


class TestSchemaIsStable:
    def test_a_batch_with_no_elements_writes_the_same_columns_as_one_with_some(self) -> None:
        """Two Parquet files in one dataset that disagree on columns make a
        reader drop whichever columns are not in both."""
        full = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(_paper()))
        empty = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch([_metadata_row()]))

        assert empty.to_pyarrow().schema.names == full.to_pyarrow().schema.names


# ---- a rejoined paragraph must not claim geometry it does not have ----------


class TestGeometry:
    def test_a_paragraph_rejoined_across_a_page_has_no_bounding_box(self) -> None:
        """It keeps the page it started on and the bbox of the fragment that
        finished it; those describe different pieces of paper."""
        records = [
            _metadata_row(),
            _element_row(0, "Text", "a sentence that runs on", page=0, bbox=LEFT_COLUMN),
            _element_row(1, "Text", "and finishes.", page=1, bbox=RIGHT_COLUMN),
        ]

        joined = next(
            r
            for r in _rows(_run(NemotronParseMarkdownPostprocessor().decompose(), _batch(records)))
            if r["element_class"] == "Text"
        )

        assert json.loads(joined["source_positions"]) == [0, 1]
        assert json.loads(joined["source_ref"]) == {"page": 0, "bbox": None}

    def test_a_paragraph_rejoined_within_a_page_keeps_one(self) -> None:
        records = [
            _metadata_row(),
            _element_row(0, "Text", "a sentence that runs on", page=0, bbox=LEFT_COLUMN),
            _element_row(1, "Text", "and finishes.", page=0, bbox=RIGHT_COLUMN),
        ]

        joined = next(
            r
            for r in _rows(_run(NemotronParseMarkdownPostprocessor().decompose(), _batch(records)))
            if r["element_class"] == "Text"
        )

        assert json.loads(joined["source_ref"])["bbox"] is not None


# ---- the six stages and the fused stage must not drift apart ----------------


class TestFusedAgreesWithStaged:
    @pytest.mark.parametrize(
        "config",
        [
            Config(),
            Config(keep_images=False),
            Config(assign_floats=False),
            Config(reconstitute_paragraphs=False),
            Config(skip_toc_bib=False),
            Config(drop_page_furniture=False),
            Config(strip_markdown=True),
            Config(drop_classes=frozenset({"Footnote"})),
        ],
    )
    @pytest.mark.parametrize("emit_dropped", [False, True])
    def test_the_two_shapes_produce_the_same_table(self, config: Config, emit_dropped: bool) -> None:
        records = _paper()

        staged = _run(
            NemotronParseMarkdownPostprocessor(config=config, emit_dropped=emit_dropped).decompose(),
            _batch(records),
        )
        fused = _run(
            NemotronParseMarkdownPostprocessor(config=config, emit_dropped=emit_dropped, fuse=True).decompose(),
            _batch(records),
        )

        assert staged.to_pyarrow().schema.names == fused.to_pyarrow().schema.names
        assert _rows(staged) == _rows(fused)


# ---- several documents in one batch -----------------------------------------


class TestManyDocuments:
    def test_documents_are_kept_apart_and_in_input_order(self) -> None:
        records = [
            _metadata_row("a"),
            _element_row(0, "Text", "First paper.", sample_id="a"),
            _metadata_row("b"),
            _element_row(0, "Text", "Second paper.", sample_id="b"),
        ]

        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch(records))

        assert [r["sample_id"] for r in _rows(result)] == ["a", "a", "b", "b"]
        assert result.num_items == 2

    def test_a_document_with_nothing_but_a_metadata_row_survives(self) -> None:
        result = _run(NemotronParseMarkdownPostprocessor().decompose(), _batch([_metadata_row("a")]))

        assert [r["position"] for r in _rows(result)] == [-1]


# ---- the intermediate is inspectable, which is why the stages are separate ---


class TestIntermediate:
    def test_each_stage_leaves_its_verdicts_where_the_next_one_can_read_them(self) -> None:
        task = _batch(_paper())
        config = Config()

        after_clean = ElementCleaningStage(config=config).process(task)
        assert {rows_module.KEEP, rows_module.DROP_REASON, rows_module.SOURCE_POSITIONS} <= set(
            after_clean.to_pyarrow().schema.names
        )

        after_floats = FloatAssignmentStage(config=config).process(after_clean)
        slots = [r[rows_module.SLOT] for r in _rows(after_floats) if r["position"] >= 0]
        assert set(slots) == {rows_module.FLOW, rows_module.FLOAT}

        after_furniture = PageFurnitureStage(config=config).process(after_floats)
        footer = next(r for r in _rows(after_furniture) if r["element_class"] == "Page-footer")
        assert (footer[rows_module.KEEP], footer[rows_module.DROP_REASON]) == (False, "page-furniture")

        after_sections = SectionSkippingStage(config=config).process(after_furniture)
        reference = next(r for r in _rows(after_sections) if "Phys. Rev." in (r["text_content"] or ""))
        assert reference[rows_module.DROP_REASON] == "toc-bibliography"

    def test_reconstitution_is_the_one_stage_that_emits_fewer_rows(self) -> None:
        config = Config()
        stages = [
            ElementCleaningStage(config=config),
            FloatAssignmentStage(config=config),
            PageFurnitureStage(config=config),
            SectionSkippingStage(config=config),
        ]
        before = _run(stages, _batch(_paper()))

        after = ParagraphReconstitutionStage(config=config).process(before)

        assert len(_rows(after)) == len(_rows(before)) - 1
