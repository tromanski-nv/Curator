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

"""Tests for the post-processing rules.

Nothing here touches Curator, Ray or Arrow: the steps are pure functions over
dataclasses and are tested as such.  The stages that wrap them are tested in
``test_stages.py``.

The inputs are shapes taken from the Nemotron-Parse corpus, and several of the
cases below are the incidents that put the rules there in the first place --
the page number in the middle of a sentence, the appendix the model labelled as
part of the bibliography, the caption that was really a stray line of layout.
"""

from __future__ import annotations

import pytest

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing import steps
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import (
    Config,
    Element,
)
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.text import (
    is_to_be_continued,
    strip_markdown,
)

#: Wide enough to overlap anything else on the page, i.e. one column.
FULL_WIDTH = (0.1, 0.1, 0.9, 0.2)
LEFT_COLUMN = (0.1, 0.1, 0.45, 0.9)
RIGHT_COLUMN = (0.55, 0.1, 0.9, 0.9)


def _el(  # noqa: PLR0913 -- an element factory mirrors the fields of an element
    position: int,
    element_class: str,
    text: str,
    page: int | None = 0,
    bbox: tuple[float, float, float, float] | None = FULL_WIDTH,
    *,
    has_payload: bool = False,
) -> Element:
    """One element, with the fields a test does not care about defaulted."""
    modality = "image" if element_class == "Picture" else "text"
    return Element(
        position=position,
        modality=modality,
        element_class=element_class,
        text=text,
        page=page,
        bbox=bbox,
        has_payload=has_payload,
    )


# ---- the text helpers, pinned hard: they decide where paragraphs join ------


class TestIsToBeContinued:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("A complete sentence.", False),
            ("A question?", False),
            ("An exclamation!", False),
            ("this runs on", True),
            ("", False),
            # Bare digits and commas AFTER a terminator are reference markers --
            # "ended. 12" is finished, the 12 is a citation number.  Only bare
            # digits/commas/space are skipped, so a closing bracket still counts
            # as content and keeps the box open.
            ("ended. 12", False),
            ("ended.12, 34", False),
            ("ended.   ", False),
            ("as shown in [12], 34", True),
            ("no terminator at all", True),
            ("ends with a comma,", True),
            ("ends with a colon:", True),
        ],
    )
    def test_terminal_punctuation_closes_a_paragraph(self, text: str, expected: bool) -> None:
        assert is_to_be_continued(text) is expected


class TestStripMarkdown:
    def test_bold_is_stripped_before_italic(self) -> None:
        """Italic first would eat one asterisk of each bold delimiter."""
        assert strip_markdown("**bold** and *italic*") == "bold and italic"

    def test_headings_links_and_leaders(self) -> None:
        assert strip_markdown("# Heading") == "Heading"
        assert strip_markdown("[text](http://x)") == "text"
        assert strip_markdown("a . . . . . . . . b").count(".") <= 5


# ---- the behaviour that motivated the port --------------------------------


class TestParagraphReconstitution:
    def test_a_sentence_split_by_a_page_break_is_reunited_over_the_page_number(self) -> None:
        """The case from 2410.10730: page 1 ends mid-sentence, a Page-header
        carrying the page number sits between, and page 2 continues it."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "We propose a model for a quantum emitter interacting", page=0),
                _el(1, "Page-header", "2", page=1),
                _el(2, "Text", "with a dispersive dielectric object based on the formalism.", page=1),
            ]
        )

        assert "interacting with a dispersive" in doc.text
        assert "2" not in doc.text.split("interacting")[1][:20]

        furniture = [b for b in doc.boxes if b.element_class == "Page-header"]
        assert len(furniture) == 1
        assert furniture[0].keep is False
        assert furniture[0].reason == "page-furniture"
        assert furniture[0].text == "2"

    def test_a_joined_box_records_which_elements_it_came_from(self) -> None:
        """``source_positions`` is what lets a viewer show the original boundaries."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "First half runs on", page=0),
                _el(1, "Page-header", "7", page=1),
                _el(2, "Text", "and the second half completes it.", page=1),
            ]
        )

        joined = [b for b in doc.boxes if b.is_joined]
        assert len(joined) == 1
        assert joined[0].source_positions == (0, 2)
        assert doc.stats.paragraphs_reconstituted == 1

    def test_a_completed_sentence_is_not_joined_to_the_next(self) -> None:
        doc = steps.postprocess(
            [
                _el(0, "Text", "A complete thought.", page=0),
                _el(1, "Text", "A separate thought.", page=1),
            ]
        )

        assert all(not b.is_joined for b in doc.boxes)
        assert doc.text == "A complete thought.\n\nA separate thought."

    def test_boxes_in_one_column_are_not_joined(self) -> None:
        """Overlapping x means an ordinary paragraph break, not a layout artefact."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on", bbox=(0.1, 0.1, 0.5, 0.2)),
                _el(1, "Text", "more text", bbox=(0.1, 0.3, 0.5, 0.4)),
            ]
        )

        assert all(not b.is_joined for b in doc.boxes)

    def test_boxes_in_different_columns_are_joined(self) -> None:
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on", bbox=LEFT_COLUMN),
                _el(1, "Text", "continued here.", bbox=RIGHT_COLUMN),
            ]
        )

        assert any(b.is_joined for b in doc.boxes)

    def test_a_section_header_flushes_an_open_paragraph(self) -> None:
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on"),
                _el(1, "Section-header", "II. Methods"),
                _el(2, "Text", "new paragraph."),
            ]
        )

        assert all(not b.is_joined for b in doc.boxes)

    def test_a_condemned_box_takes_the_half_paragraph_before_it_with_it(self) -> None:
        """The continuation the open half was waiting for will never arrive."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on", bbox=LEFT_COLUMN),
                _el(1, "Text", "word " * 150, bbox=RIGHT_COLUMN),
            ]
        )

        first = next(b for b in doc.boxes if b.source_positions == (0,))
        assert first.keep is False
        assert first.reason == "joined-into-previous"

    def test_a_skipped_section_ends_the_paragraph_without_condemning_it(self) -> None:
        """A bibliography after a half-written sentence is a document that
        stopped, not a sentence that broke."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on", bbox=LEFT_COLUMN),
                _el(1, "Bibliography", "[1] Someone.", bbox=RIGHT_COLUMN),
            ]
        )

        first = next(b for b in doc.boxes if b.source_positions == (0,))
        assert first.keep is True


# ---- drops are marked, never removed ---------------------------------------


class TestDropsAreMarked:
    def test_a_short_caption_is_marked_not_deleted(self) -> None:
        doc = steps.postprocess([_el(0, "Picture", ""), _el(1, "Caption", "Fig 1.")])

        caption = [b for b in doc.boxes if b.element_class == "Caption"]
        assert len(caption) == 1
        assert caption[0].keep is False
        assert caption[0].reason == "short-caption"

    def test_a_table_without_a_tabular_environment_is_marked(self) -> None:
        doc = steps.postprocess([_el(0, "Table", "just some prose, no environment")])
        assert doc.boxes[0].reason == "table-without-tabular"

        kept = steps.postprocess([_el(0, "Table", r"\begin{tabular}a\end{tabular}")])
        assert kept.boxes[0].keep is True

    def test_every_input_element_is_represented_in_the_output(self) -> None:
        """The core promise: nothing is dropped from ``boxes``, only from ``text``."""
        elements = [
            _el(0, "Title", "A Paper", page=0),
            _el(1, "Text", "Body runs on", page=0),
            _el(2, "Page-header", "1", page=1),
            _el(3, "Text", "and finishes.", page=1),
            _el(4, "Caption", "short", page=1),
            _el(5, "Bibliography", "[1] Someone.", page=1),
        ]

        doc = steps.postprocess(elements)

        covered = {p for b in doc.boxes for p in b.source_positions}
        assert covered == {e.position for e in elements}

    def test_a_bibliography_is_left_once_real_prose_resumes(self) -> None:
        """The model labels an appendix following the references as part of
        them, so the skip has to end on content rather than on a heading."""
        prose = (
            "A long paragraph of genuine prose with many words in it that clears the "
            "threshold because it has well over thirty five real words in it and keeps "
            "going on and on and on for a good while yet without stopping anywhere."
        )
        doc = steps.postprocess(
            [
                _el(0, "Section-header", "References"),
                _el(1, "Text", "[1] A. Someone, Phys. Rev. A 12, 345 (2020)."),
                _el(2, "Text", prose),
            ]
        )

        assert doc.boxes[1].reason == "toc-bibliography"
        assert doc.boxes[2].keep is True


# ---- floats -----------------------------------------------------------------


class TestFloats:
    def test_a_figure_is_followed_by_the_caption_matched_to_it(self) -> None:
        caption = "FIG. 1. " + "a genuinely long caption with plenty of words in it " * 3
        layout = steps.assign_floats(
            steps.clean(
                [
                    _el(0, "Text", "Body text.", bbox=(0.1, 0.05, 0.9, 0.10)),
                    _el(1, "Caption", caption, bbox=(0.1, 0.60, 0.9, 0.65)),
                    _el(2, "Picture", "", bbox=(0.1, 0.30, 0.9, 0.55), has_payload=True),
                ],
                Config(),
            ),
            Config(),
        )

        assert [b.element_class for b in layout[0].flow] == ["Text"]
        assert [b.element_class for b in layout[0].floats] == ["Picture", "Caption"]
        assert layout[0].floats[1].matched_to == 2

    def test_prose_joins_across_an_intervening_figure(self) -> None:
        """A figure dropped into the middle of a column must not cut a sentence."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on", bbox=LEFT_COLUMN),
                _el(1, "Picture", "", bbox=(0.4, 0.4, 0.6, 0.6), has_payload=True),
                _el(2, "Text", "and finishes.", bbox=RIGHT_COLUMN),
            ]
        )

        assert "runs on and finishes." in doc.text

    def test_switching_float_assignment_off_leaves_them_in_the_flow(self) -> None:
        cfg = Config(assign_floats=False)
        layout = steps.assign_floats(
            steps.clean([_el(0, "Text", "Body."), _el(1, "Picture", "", has_payload=True)], cfg), cfg
        )

        assert layout[0].floats == ()
        assert [b.element_class for b in layout[0].flow] == ["Text", "Picture"]


# ---- images -----------------------------------------------------------------


class TestImages:
    def test_a_picture_with_no_text_survives_because_its_payload_is_not_text(self) -> None:
        """The interleaved default.  Judged by its text alone a picture is
        empty, and dropping it would leave a text-only corpus."""
        doc = steps.postprocess([_el(0, "Picture", "", has_payload=True)])

        assert doc.boxes[0].keep is True

    def test_the_text_only_baseline_still_drops_it(self) -> None:
        doc = steps.postprocess([_el(0, "Picture", "", has_payload=True)], Config(keep_images=False))

        assert doc.boxes[0].keep is False
        assert doc.boxes[0].reason == "empty"


# ---- the steps compose into the whole -------------------------------------


class TestComposition:
    def test_running_the_steps_by_hand_gives_what_postprocess_gives(self) -> None:
        """``postprocess`` is exactly the six steps in order, and the stages in
        ``stages.py`` call them one at a time.  If this drifts, a fused run and
        a staged run stop agreeing."""
        elements = [
            _el(0, "Title", "A Paper", page=0),
            _el(1, "Text", "Body runs on", page=0, bbox=LEFT_COLUMN),
            _el(2, "Picture", "", page=0, bbox=(0.4, 0.4, 0.6, 0.6), has_payload=True),
            _el(3, "Page-header", "2", page=1),
            _el(4, "Text", "and finishes.", page=1, bbox=RIGHT_COLUMN),
            _el(5, "Caption", "short", page=1),
            _el(6, "Bibliography", "[1] Someone.", page=1),
        ]
        cfg = Config()

        layout = steps.assign_floats(steps.clean(elements, cfg), cfg)
        layout = steps.mark_page_furniture(layout, cfg)
        layout = steps.skip_sections(layout, cfg)
        layout = steps.reconstitute_paragraphs(layout, cfg)
        by_hand = steps.assemble(layout)

        assert by_hand == steps.postprocess(elements, cfg).boxes
        assert steps.summarise(by_hand) == steps.postprocess(elements, cfg).stats

    def test_every_flag_can_be_switched_off_independently(self) -> None:
        elements = [
            _el(0, "Text", "runs on", page=0),
            _el(1, "Page-header", "2", page=1),
            _el(2, "Text", "and completes.", page=1),
        ]

        raw = steps.postprocess(elements, Config(reconstitute_paragraphs=False, drop_page_furniture=False))

        assert all(not b.is_joined for b in raw.boxes)
        assert "2" in raw.text

    def test_postprocess_does_not_mutate_its_input(self) -> None:
        elements = [_el(0, "Text", "**bold** runs on"), _el(1, "Text", "done.")]
        before = [(e.position, e.text, e.element_class) for e in elements]

        steps.postprocess(elements)

        assert [(e.position, e.text, e.element_class) for e in elements] == before

    def test_an_empty_document(self) -> None:
        doc = steps.postprocess([])

        assert doc.boxes == ()
        assert doc.text == ""

    def test_elements_with_no_page_are_one_synthetic_page(self) -> None:
        """A page number is optional, so the rules need a total order without
        one.  Everything unpaged is one page, and joins across it as usual."""
        doc = steps.postprocess(
            [
                _el(0, "Text", "runs on", page=None, bbox=LEFT_COLUMN),
                _el(1, "Text", "and ends.", page=None, bbox=RIGHT_COLUMN),
            ]
        )

        assert len(doc.boxes) == 1
        assert doc.boxes[0].page is None
        assert doc.boxes[0].text == "runs on and ends."
