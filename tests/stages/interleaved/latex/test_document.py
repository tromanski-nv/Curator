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

"""convert_submission: one submission in, one ConvertedDocument out.

These cover the failure modes the corpus run surfaced, each of which was a
silent data defect rather than a crash.
"""

import gzip

import pytest

from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import LatexmlConfig
from nemo_curator.stages.interleaved.latex.arxiv.latexml.document import (
    arxiv_id_from_member,
    convert_submission,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.model import Status, Tier

from .conftest import LATEX_SOURCE

SHARD = "arXiv_src_0301_001.tar"


def _convert(tmp_path, member, payload, **kwargs):
    return convert_submission(payload, member, SHARD, "d" * 64, tmp_path / "work", **kwargs)


# --- identifiers ---


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        pytest.param("0301/astro-ph0301029.gz", "astro-ph/0301029", id="old_style_gets_a_slash"),
        pytest.param("1203/1203.5560.gz", "1203.5560", id="new_style_unchanged"),
        # A PDF-only submission once kept its extension, giving every such row a
        # URL that 404s -- 100% of pdf_only rows, ~210k corpus-wide.
        pytest.param("1203/1203.5560.pdf", "1203.5560", id="pdf_suffix_stripped"),
        pytest.param("9403/hep-th9403001.gz", "hep-th/9403001", id="hyphenated_archive"),
    ],
)
def test_arxiv_id_from_member(member, expected):
    assert arxiv_id_from_member(SHARD, member) == expected


# --- submission shapes ---


def test_pdf_only_submission_is_counted_not_converted(tmp_path, latexmlc):
    """No LaTeX exists, but the row must still appear or the denominator shifts."""
    doc = _convert(tmp_path, "1203/1203.5560.pdf", b"%PDF-1.4")
    assert doc.kind == "pdf_only"
    assert doc.html is None
    assert doc.status is Status.NO_SOURCE
    assert doc.tier is Tier.REJECTED
    # kind carries the detail; writing it into status polluted the vocabulary.
    assert doc.status.value not in {"pdf_only", "tar", "single_file", "empty"}


def test_submission_with_no_root_tex_is_counted(tmp_path, latexmlc):
    """A project tarball carrying no .tex at all still produces a row."""
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("figure.png")
        info.size = 4
        tf.addfile(info, io.BytesIO(b"\x89PNG"))
    doc = _convert(tmp_path, "0301/x.gz", gzip.compress(buf.getvalue()))
    assert doc.root_tex is None
    assert doc.html is None
    assert doc.tier is Tier.REJECTED


def test_successful_conversion_is_graded_and_counted(tmp_path, latexmlc):
    doc = _convert(tmp_path, "0301/astro-ph0301029.gz", gzip.compress(LATEX_SOURCE))
    assert doc.converted
    assert doc.arxiv_id == "astro-ph/0301029"
    assert doc.tier in {Tier.A, Tier.B}
    assert doc.counts.n_math == 1
    assert doc.counts.n_alttext == 1
    assert doc.counts.n_img == 1
    assert doc.counts.n_section == 2
    assert doc.source_expects_math is True
    assert doc.source_expects_figures is True


def test_oversized_output_is_rejected_before_it_is_kept(tmp_path, latexmlc):
    """A runaway must be dropped while it is still one string.

    One paper produced 1,444 MB and drove peak RSS to 10.2 GB. Row and byte
    limits cannot catch it: they are tested after the row is appended.
    """
    doc = _convert(tmp_path, "0301/astro-ph0301029.gz", gzip.compress(LATEX_SOURCE), max_html_bytes=64)
    assert doc.status is Status.SUSPECT_OVERSIZED
    assert doc.tier is Tier.REJECTED
    assert doc.failed_gates == ("oversized",)
    assert doc.html is None, "the oversized HTML must not be carried in the row"


def test_source_expectations_are_null_when_the_source_was_never_read(tmp_path, latexmlc):
    """NULL and False are different facts.

    An unreadable source that graded as "no math" would disable the one gate
    that catches wholesale math deletion, grading the document better than it is.
    """
    doc = _convert(tmp_path, "1203/1203.5560.pdf", b"%PDF-1.4")
    assert doc.source_expects_math is None
    assert doc.source_expects_figures is None


# --- scratch and assets ---


def test_scratch_is_removed_even_though_the_document_converted(tmp_path, latexmlc):
    workdir = tmp_path / "work"
    _convert(tmp_path, "0301/astro-ph0301029.gz", gzip.compress(LATEX_SOURCE))
    assert list(workdir.iterdir()) == [], "unpacked project must not outlive the conversion"


def test_asset_sink_is_called_while_the_files_still_exist(tmp_path, latexmlc):
    latexmlc(body="<html><body><p>" + ("word " * 200) + "</p></body></html>")
    seen = []

    def sink(path, arcname):
        assert path.exists(), "assets must be drained before the scratch tree is removed"
        seen.append(arcname)

    doc = _convert(tmp_path, "0301/astro-ph0301029.gz", gzip.compress(LATEX_SOURCE), asset_sink=sink)
    assert doc.n_assets == len(seen)


# --- config is honoured ---


def test_conversion_uses_the_supplied_config(tmp_path, latexmlc):
    """A non-default executable must actually be the one invoked."""
    doc = _convert(
        tmp_path,
        "0301/astro-ph0301029.gz",
        gzip.compress(LATEX_SOURCE),
        config=LatexmlConfig(executable="definitely-not-a-real-binary"),
    )
    # The converter could not be launched, but no exception escaped: one bad
    # document must not end a shard.
    assert doc.html is None
    assert doc.tier is Tier.REJECTED
    assert "launch_failed" in (doc.log or "")
