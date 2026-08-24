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

r"""Tests for the markdown rendering step.

The inputs below are shapes taken verbatim from the Nemotron-Parse corpus.
``strip_markdown`` is off by default, so the plain ``_md(...)`` calls exercise
the path the corpus actually takes; :data:`STRIPPED` is the predecessor
pipeline's plain-text baseline, kept so the two stay comparable.
"""

from __future__ import annotations

from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.markdown import to_markdown
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.model import Config, Element
from nemo_curator.stages.interleaved.pdf.nemotron_parse.postprocessing.steps import postprocess

#: The predecessor pipeline's baseline: markdown reduced to plain text.
STRIPPED = Config(strip_markdown=True)


def _el(
    position: int,
    element_class: str,
    text: str,
    page: int = 0,
    bbox: tuple[float, float, float, float] = (0.1, 0.1, 0.9, 0.2),
) -> Element:
    return Element(
        position=position,
        modality="text",
        element_class=element_class,
        text=text,
        page=page,
        bbox=bbox,
    )


def _md(elements: list[Element], cfg: Config | None = None) -> str:
    return to_markdown(postprocess(elements, cfg).boxes)


# ---- headings ---------------------------------------------------------------


class TestHeadings:
    def test_existing_hashes_are_not_doubled(self) -> None:
        out = _md([_el(0, "Title", "# A Paper"), _el(1, "Section-header", "## II. MODEL")])
        assert out == "# A Paper\n\n## II. MODEL"

    def test_hashes_are_added_when_stripping_removed_them(self) -> None:
        out = _md([_el(0, "Title", "# A Paper"), _el(1, "Section-header", "## II. MODEL")], STRIPPED)
        assert out == "# A Paper\n\n## II. MODEL"

    def test_the_corpus_heading_level_is_preserved(self) -> None:
        """``###`` under ``##`` is real nesting; flattening it loses the outline."""
        out = _md([_el(0, "Section-header", "### C.2 ATTENTION SINK"), _el(1, "Section-header", "## VI.")])
        assert out == "### C.2 ATTENTION SINK\n\n## VI."

    def test_an_unhashed_heading_gets_hashes(self) -> None:
        out = _md([_el(0, "Title", "A Paper"), _el(1, "Section-header", "ABSTRACT")])
        assert out == "# A Paper\n\n## ABSTRACT"

    def test_hashes_with_no_space_are_separated(self) -> None:
        assert _md([_el(0, "Section-header", "###MLI over permutation symmetries.")]) == (
            "### MLI over permutation symmetries."
        )

    def test_a_heading_broken_over_two_lines_is_collapsed(self) -> None:
        """Markdown would end the heading at the newline and strand the rest."""
        out = _md([_el(0, "Title", "# Super-resolution of ultrafast pulses via spectral\ninversion")])
        assert out == "# Super-resolution of ultrafast pulses via spectral inversion"


# ---- math: inline vs display comes from the element class -------------------


class TestMath:
    def test_math_inside_prose_is_inline(self) -> None:
        out = _md([_el(0, "Text", r"The field \(\hat{E}(r)\) has two parts.")])
        assert out == r"The field $\hat{E}(r)$ has two parts."

    def test_a_formula_element_is_display_math(self) -> None:
        out = _md([_el(0, "Formula", r"\(\hat{H}=\hat{H}_a+\hat{H}_I,\) (1)")])
        assert out == r"$$\hat{H}=\hat{H}_a+\hat{H}_I,$$ (1)"

    def test_the_equation_number_stays_outside_the_math(self) -> None:
        out = _md([_el(0, "Formula", r"\(x=1\) (14)")])
        assert out.endswith("$$ (14)")

    def test_escaped_brackets_are_citations_not_display_math(self) -> None:
        r"""10.7% of Text carries ``\[...\]``; reading it as math destroys citations."""
        out = _md([_el(0, "Text", r"as formulated in \[28\]. The operator \(\hat{E}\) is.")])
        assert r"\[28\]" in out
        assert out.count("$") == 2

    def test_a_span_crossing_newlines_is_converted(self) -> None:
        r"""4.5% of formulas wrap a ``\begin{array}`` and so span lines."""
        src = "\\(f(t)=\\left\\{\\begin{array}{cc}\nt & a=0 \\\\\nu & a>0\n\\end{array}\\right.\\) (11)"
        out = _md([_el(0, "Formula", src)])
        assert out.startswith("$$f(t)=")
        assert out.endswith("$$ (11)")
        assert "\\(" not in out

    def test_an_unpaired_delimiter_is_left_exactly_as_it_was(self) -> None:
        """0.3% of formulas never close a span.  The orphan must not pair with
        the next span's closer, which would eat the delimiter between them."""
        out = _md([_el(0, "Formula", "\\(\\mathcal{C}(\\alpha\n\\(+(1-\\eta)\\) (10)")])
        assert out == "\\(\\mathcal{C}(\\alpha\n$$+(1-\\eta)$$ (10)"

    def test_a_literal_dollar_is_escaped_before_math_is_introduced(self) -> None:
        """Unescaped, it would pair with a delimiter we add and swallow the prose."""
        out = _md([_el(0, "Text", r"it costs $5, and \(x=1\) holds")])
        assert out == r"it costs \$5, and $x=1$ holds"

    def test_an_empty_span_is_deleted_rather_than_delimited(self) -> None:
        r"""The model writes ``\(\)`` where it saw maths it could not read.
        ``$$`` is not an empty formula to a markdown reader -- it is an opening
        display-math delimiter, and it swallows the document after it."""
        out = _md([_el(0, "Text", r"The bracket notation \(\) stands for the expected value.")])

        assert "$" not in out
        assert out.startswith("The bracket notation")
        assert out.endswith("stands for the expected value.")

    def test_an_empty_span_beside_a_real_one_leaves_the_real_one_alone(self) -> None:
        out = _md([_el(0, "Text", r"with \( \) and real \(x=1\) together")])

        assert out == "with  and real $x=1$ together"

    def test_math_in_a_table_is_left_as_latex(self) -> None:
        table = "\\begin{tabular}{cc}\n\\(x\\) & b \\\\\n\\end{tabular}"
        assert _md([_el(0, "Table", table)]) == table

    def test_math_in_code_is_left_alone(self) -> None:
        r"""Inside a fence ``$`` and ``\(`` are already literal."""
        out = _md([_el(0, "Code", "Initialize \\(A\\leftarrow\\emptyset\\)")])
        assert out == "```\nInitialize \\(A\\leftarrow\\emptyset\\)\n```"


# ---- tables stay LaTeX ------------------------------------------------------


class TestTables:
    def test_a_table_is_emitted_as_latex_untouched(self) -> None:
        table = "\\begin{tabular}{cc}\na & **b** \\\\\n\\end{tabular}"
        out = _md([_el(0, "Table", table)])
        assert out == table
        assert "|" not in out

    def test_a_table_without_a_tabular_is_dropped_before_rendering(self) -> None:
        """The renderer shows what the rules kept, and nothing else."""
        assert _md([_el(0, "Table", "prose, no environment")]) == ""


# ---- list items -------------------------------------------------------------


class TestListItems:
    def test_a_unicode_bullet_becomes_a_markdown_bullet(self) -> None:
        assert _md([_el(0, "List-item", "• Capabilities are portable.")]) == "- Capabilities are portable."

    def test_an_unmarked_list_item_gets_a_bullet(self) -> None:
        assert _md([_el(0, "List-item", "This follows.")]) == "- This follows."

    def test_an_existing_markdown_marker_is_left_alone(self) -> None:
        assert _md([_el(0, "List-item", "3. This follows from (1).")]) == "3. This follows from (1)."

    def test_a_math_wrapped_bullet_does_not_become_a_double_bullet(self) -> None:
        out = _md([_el(0, "List-item", r"\(\bullet\) We introduce an approach.")])
        assert out == "- We introduce an approach."

    def test_a_math_wrapped_bullet_leaves_real_math_balanced(self) -> None:
        """When the span continues into real math, only the bullet goes."""
        out = _md([_el(0, "List-item", r"\(\bullet s_0-s_1<0\) holds")])
        assert out == "- $s_0-s_1<0$ holds"

    def test_an_unrecognised_label_keeps_its_label(self) -> None:
        """``(a)`` is not markdown, so it gets a bullet rather than being parsed."""
        assert _md([_el(0, "List-item", "(a) the first case")]) == "- (a) the first case"

    def test_math_is_converted_in_an_already_marked_list_item(self) -> None:
        assert _md([_el(0, "List-item", r"3. because \(x=1\).")]) == r"3. because $x=1$."


# ---- everything else --------------------------------------------------------


class TestOtherClasses:
    def test_a_paragraph_starting_with_a_hash_is_escaped(self) -> None:
        """A code comment caught inside a figure; bare it would open a section."""
        out = _md([_el(0, "Text", "# Calculate the projection matrix")])
        assert out == "\\# Calculate the projection matrix"

    def test_code_is_fenced(self) -> None:
        assert _md([_el(0, "Code", "while x:\n  step()")]) == "```\nwhile x:\n  step()\n```"

    def test_code_containing_a_fence_gets_a_longer_one(self) -> None:
        out = _md([_el(0, "Code", "print('```')")])
        assert out.startswith("````\n")
        assert out.endswith("\n````")

    def test_a_picture_carries_no_text_and_renders_to_nothing(self) -> None:
        assert _md([_el(0, "Picture", ""), _el(1, "Text", "Body.")]) == "Body."

    def test_an_unknown_element_class_becomes_a_paragraph(self) -> None:
        """A class a later release adds degrades to readable text, not to nothing."""
        assert _md([_el(0, "Neologism", "Some text.")]) == "Some text."


# ---- the same selection as .text --------------------------------------------


class TestSelection:
    def test_dropped_boxes_do_not_reach_the_markdown(self) -> None:
        elements = [
            _el(0, "Title", "# A Paper"),
            _el(1, "Page-header", "2", page=1),
            _el(2, "Caption", "short", page=1),
            _el(3, "Text", "Body.", page=1),
        ]

        assert _md(elements) == "# A Paper\n\nBody."
        assert len(postprocess(elements).boxes) == 4  # ...but all four are recorded

    def test_an_empty_document(self) -> None:
        assert to_markdown(postprocess([]).boxes) == ""
