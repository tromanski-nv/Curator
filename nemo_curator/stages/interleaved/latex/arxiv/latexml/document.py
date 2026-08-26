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

r"""One submission in, one :class:`ConvertedDocument` out.

The whole per-document pipeline -- unpack, pick a root, convert, grade, scan --
with no Curator types and no knowledge of where the submission came from or
where the result is going.  :mod:`..latexml.stage` supplies both.

It never raises for a document that fails to convert.  A bad paper is a row
with a status, not an exception: one unconvertible submission out of 12,830
shards must not end a job.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from pathlib import Path

from nemo_curator.stages.interleaved.latex.arxiv.latexml.artifacts import scan as scan_artifacts
from nemo_curator.stages.interleaved.latex.arxiv.latexml.boilerplate import strip_boilerplate
from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import (
    AR5IV_CONFIG,
    LatexmlConfig,
    convert,
    error_kinds,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.extract import extract_submission
from nemo_curator.stages.interleaved.latex.arxiv.latexml.model import (
    MAX_HTML_BYTES,
    ConvertedDocument,
    Status,
    Tier,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.quality import (
    assess,
    source_expects_figures,
    source_expects_math,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.source_text import strip_comments

_OLD_ID_RE = re.compile(r"^([a-zA-Z-]+)(\d{7})$")


def shard_stem(shard: str) -> str:
    """``arXiv_src_0301_001.tar`` -> ``arXiv_src_0301_001``."""
    return shard.removesuffix(".tar")


def arxiv_id_from_member(shard: str, member: str) -> str:
    r"""Derive the arXiv id from a shard name and a tar member name.

    ``arXiv_src_0301_001.tar`` + ``astro-ph0301029.gz`` -> ``astro-ph/0301029``.

    The ``.pdf`` strip is not cosmetic.  PDF-only submissions once kept the
    extension in their identifier, so every such row carried
    ``arxiv_id='1203.5560.pdf'`` and a URL that 404s -- measured at 100% of
    ``pdf_only`` rows, ~210k corpus-wide.  Those are exactly the documents a PDF
    fallback would need to join back to arXiv by id.

    There is deliberately one place that derives an identifier, and the caller
    passes the *whole* member name including its extension: stripping ``.gz``
    before calling is what produced the bug above.
    """
    name = f"{shard_stem(shard)}__{Path(member).name}"
    stem = name.split("__", 1)[-1].removesuffix(".gz").removesuffix(".pdf")
    match = _OLD_ID_RE.match(stem)
    return f"{match.group(1)}/{match.group(2)}" if match else stem


def convert_submission(  # noqa: PLR0911 -- each early return is a distinct, documented submission shape
    payload: bytes,
    member: str,
    shard: str,
    digest: str,
    workdir: Path,
    *,
    config: LatexmlConfig = AR5IV_CONFIG,
    max_html_bytes: int = MAX_HTML_BYTES,
    asset_sink: Callable[[Path, str], None] | None = None,
) -> ConvertedDocument:
    """Unpack, convert and grade one tar member.

    *workdir* is scratch: the submission is unpacked into it and the unpacked
    tree is deleted before returning, whatever happens.  That is what bounds
    staging to the number of documents in flight rather than to a whole shard.

    *asset_sink* is called as ``sink(path, arcname)`` for each rasterized figure
    LaTeXML produced, while the file still exists.  Assets cannot be returned:
    the scratch tree is gone by the time this function does.
    """
    arxiv_id = arxiv_id_from_member(shard, member)
    project_dir = workdir / Path(member).name.removesuffix(".gz")
    doc = lambda **kw: ConvertedDocument(arxiv_id=arxiv_id, source_sha256=digest, **kw)  # noqa: E731

    try:
        if member.endswith(".pdf"):
            # No LaTeX exists.  Counted so the denominator stays "of all
            # submissions".  ``kind`` carries the detail rather than ``status``:
            # writing it into ``status`` polluted the status vocabulary with
            # ``tar``/``empty``/``single_file`` on 0.48% of live rows, and
            # ``pdf_only`` was never a Status.  assess() already grades a
            # sourceless submission NO_SOURCE, so the default agrees with it.
            return doc(kind="pdf_only")

        project = extract_submission(payload, member, project_dir)
        if not project.root_tex:
            return doc(kind=project.kind)

        # Read before converting: an unreadable root .tex is a defect either way,
        # and finding out first saves ~40 core-seconds of pointless conversion.
        # ``errors="replace"`` is deliberate (LaTeX is routinely latin-1 inside a
        # utf-8 file); the encoding is pinned because leaving it out made the
        # decode depend on the container's locale.
        try:
            source = strip_comments((project_dir / project.root_tex).read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            # Never "" on failure.  An empty source makes source_expects_math()
            # false, which silently disables the no_math gate -- the one gate
            # that catches wholesale math deletion -- so a read failure would
            # grade the document *better* than it is.
            return doc(
                kind=project.kind,
                root_tex=project.root_tex,
                status=Status.ERROR,
                tier=Tier.REJECTED,
                failed_gates=("source_unreadable",),
                log=f"Fatal:source_unreadable: {type(exc).__name__}: {exc}",
            )

        out_dir = project_dir / "_out"
        result = convert(project_dir, project.root_tex, out_dir / "index.html", config=config)
        log = result.log

        # Checked before strip_boilerplate and before anything is built from it:
        # a runaway document must be dropped while it is still one string, not
        # after it has been copied into a row, an Arrow table and a write buffer.
        if result.html is not None and len(result.html) > max_html_bytes:
            oversized = len(result.html)
            del result
            return doc(
                kind=project.kind,
                root_tex=project.root_tex,
                status=Status.SUSPECT_OVERSIZED,
                tier=Tier.REJECTED,
                failed_gates=("oversized",),
                log=f"Fatal:oversized: {oversized} bytes of HTML exceeds {max_html_bytes}\n{log}",
            )

        html = strip_boilerplate(result.html) if result.html else None
        verdict = assess(
            html,
            source,
            n_error=result.n_error,
            n_fatal=result.n_fatal,
            n_warning=result.n_warning,
            timed_out=result.timed_out,
            error_kinds=error_kinds(log),
        )
        artifacts = scan_artifacts(html) if html else None

        # Drained here, inside the try: the finally below removes the tree these
        # files live in, so a caller cannot fetch them afterwards.
        n_assets = 0
        if html:
            for asset in sorted(out_dir.glob("*.png")):
                if asset_sink is not None:
                    asset_sink(asset, f"{arxiv_id}/{asset.name}")
                n_assets += 1

        return doc(
            kind=project.kind,
            root_tex=project.root_tex,
            html=html,
            status=verdict.status,
            tier=verdict.tier,
            counts=verdict.counts,
            failed_gates=tuple(verdict.failed_gates),
            n_warning=result.n_warning,
            n_error=result.n_error,
            n_fatal=result.n_fatal,
            n_artifacts=artifacts.total if artifacts else 0,
            # Graded from the same string assess() saw, so a future change to a
            # source-dependent gate is a Parquet-only operation instead of a
            # 6.3 TB re-read of the source tars.
            source_expects_math=source_expects_math(source),
            source_expects_figures=source_expects_figures(source),
            duration_s=round(result.duration_s, 3),
            log=log,
            n_assets=n_assets,
        )
    finally:
        # Deleted here, not at the end of the shard.  Assets are read by the
        # caller before this returns, or not at all.
        shutil.rmtree(project_dir, ignore_errors=True)
