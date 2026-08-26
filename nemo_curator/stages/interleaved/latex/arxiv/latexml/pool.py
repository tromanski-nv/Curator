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

r"""The on-disk contract for the conversion pool: schema, naming, atomic writes.

Everything that two writers must agree on lives here and nowhere else.  This
module exists because they did *not* agree: the sample packer and the corpus
converter were written weeks apart and drifted on the ``status`` chosen for
unconvertible submissions, on what ``duration_s`` physically measures, and on
whether the real LaTeX source reached the quality gates at all.  Each divergence
produced a pool whose rows mean different things depending on which script wrote
them, and nothing detected any of it.

The rules encoded here, each from a measured failure:

**Part-file names are built in one place.**  Resume compares a formatted name
against files on disk; reuse globs.  When the two formats were written out
separately they could disagree, and a disagreement means silently re-converting
a finished shard or skipping an unfinished one.

**Temp names carry the writer's identity.**  They used to be a pure function of
the target, so two tasks on one shard wrote the *same* temp file and both
renamed it into place.  Reproduced: two of three trials published an unreadable
Parquet, the third silently dropped 2,500 rows.

**A published file is verified readable before it counts as done.**  Resume used
to gate on ``exists()``, so a corrupt part file was indistinguishable from a
finished one and the shard could never recover.

**The config hash covers the converter, not just its flags.**  Hashing the argv
alone let a container upgrade land in the same pool as the output it is
incompatible with, with no per-row column able to separate them afterwards.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import (
    EXCLUDED_ARGS,
    build_argv,
    converter_identity,
)

_OLD_ID_RE = re.compile(r"^([a-zA-Z-]+)(\d{7})$")

#: Rows per Parquet row group.  A part file used to be written as a single row
#: group, which made ``tier``/``status`` predicate pushdown impossible: min/max
#: spanned every value, so a tier-filtered read still decompressed 100% of the
#: ``html`` column.  Row groups are fixed at write time and cannot be changed
#: afterwards without rewriting the corpus, so this is set now or never.
ROW_GROUP_SIZE = 256

#: Largest HTML a single document may contribute.  Beyond this the output is a
#: converter pathology, and it is rejected with ``Status.SUSPECT_OVERSIZED``
#: rather than stored.
#:
#: 64 MB is 6.4x the p99.99 of 27,003 measured corpus documents (9.96 MB) and
#: rejects ~0.004% of them, roughly 240 corpus-wide.  The number that forced it
#: was 1,444 MB from one paper -- three orders of magnitude past anything
#: legitimate, and enough on its own to drive peak RSS to 10.2 GB.
#:
#: This bound, not ``FLUSH_BYTES``, is what actually caps memory.  A flush
#: threshold is tested *after* a row joins the buffer, so one row bigger than
#: the threshold blows through it no matter how the flushing is written.
MAX_HTML_BYTES = 64 * 1024 * 1024

#: Rows buffered before a part file is flushed.  Bounds peak memory to roughly
#: this many documents rather than to the size of a shard: the largest real
#: shard holds 1.68 GB of HTML, and ``Table.from_pylist`` doubles it, which is
#: what actually drove the OOMs -- not converter concurrency.
FLUSH_ROWS = 2000

#: ``frame`` distinguishes rows drawn as a *sample* (carrying a real
#: ``sample_weight``, each standing for many corpus documents) from a *census*
#: of whole shards (weight 1.0).  Without it the two are indistinguishable and
#: summing ``sample_weight`` over the pool overstates the corpus 22x -- measured
#: on the live pool, where 5% of rows carried 96% of the weight mass.
FRAME_SAMPLE = "estimation"
FRAME_CENSUS = "corpus"

SCHEMA = pa.schema(
    [
        ("arxiv_id", pa.string()),
        ("url", pa.string()),
        ("shard", pa.string()),
        ("era", pa.string()),
        ("iteration", pa.int32()),
        # Which population this row belongs to; see FRAME_* above.
        ("frame", pa.string()),
        # Source snapshot, so a later snapshot reusing this pool's conversions
        # stays separable.  Reuse is the pool's main long-term value and it is
        # one column away from being safe.
        ("snapshot", pa.string()),
        # Converter fingerprint: image digest + LaTeXML/bindings versions.
        ("converter_id", pa.string()),
        ("source_sha256", pa.string()),
        ("root_tex", pa.string()),
        ("html", pa.large_string()),
        ("status", pa.string()),
        ("tier", pa.string()),
        ("kind", pa.string()),
        ("n_warning", pa.int32()),
        ("n_error", pa.int32()),
        ("n_fatal", pa.int32()),
        ("n_math", pa.int32()),
        ("n_alttext", pa.int32()),
        ("n_img", pa.int32()),
        ("n_section", pa.int32()),
        ("n_artifacts", pa.int32()),
        # What the *source* demanded, retained so a future change to a
        # source-dependent gate (no_math, no_figures) can be re-evaluated from
        # Parquet.  Without these, re-tiering means re-reading 6.3 TB of source
        # tars -- and becomes impossible once those tars are deleted.
        ("source_expects_math", pa.bool_()),
        ("source_expects_figures", pa.bool_()),
        ("failed_gates", pa.list_(pa.string())),
        # Subprocess wall clock, always.  This column previously held converter
        # self-reported time in rows written by one script and wall clock in the
        # other, so a threshold query returned different populations for reasons
        # unrelated to the documents.
        ("duration_s", pa.float64()),
        ("sample_weight", pa.float64()),
        ("content_derivation", pa.string()),
    ]
)


def arxiv_id_from_dir(directory: str) -> str:
    r"""``arXiv_src_0301_001__astro-ph0301029`` -> ``astro-ph/0301029``.

    The ``.pdf`` strip is not cosmetic.  PDF-only submissions kept the extension
    in their identifier, so every such row carried ``arxiv_id='1203.5560.pdf'``
    and a URL that 404s -- measured at 100% of ``pdf_only`` rows, ~210k
    corpus-wide.  These are exactly the documents a PDF fallback would need to
    join back to arXiv by id.
    """
    stem = directory.split("__", 1)[-1].removesuffix(".gz").removesuffix(".pdf")
    match = _OLD_ID_RE.match(stem)
    return f"{match.group(1)}/{match.group(2)}" if match else stem


def part_name(iteration: int, part: int = 0) -> str:
    """The one definition of a part-file stem, used by resume and by writers."""
    return f"iter-{iteration:03d}-part-{part:04d}"


def shard_stem(shard: str) -> str:
    return shard.removesuffix(".tar")


def config_hash(image_path: str | None = None) -> str:
    """Identify the converter configuration, flags *and* binary.

    Hashed over the real ``build_argv`` output rather than a retyped copy of it,
    plus the converter identity (image sha256 when the pinned squashfs is
    readable, else its versions).  Hashing the argv alone meant a container
    upgrade with unchanged flags silently merged into the pool it is
    incompatible with, and ``--path=`` arguments point *into* the image, so its
    contents can change with the argv byte-identical.
    """
    argv = build_argv("<source>", "<destination>")
    identity = converter_identity(image_path)
    material = [
        *argv,
        *EXCLUDED_ARGS,
        identity.get("image_sha256", identity.get("image_path", "")),
        identity.get("latexml_version", ""),
        identity.get("ar5iv_bindings_commit", ""),
        # These bound the too_many_errors fatal, so they change output.
        identity.get("max_errors", ""),
        identity.get("max_warnings", ""),
    ]
    return hashlib.sha256("\0".join(material).encode()).hexdigest()[:12]


def pool_dir(pool_root: Path, name: str | None = None, image_path: str | None = None) -> Path:
    """Resolve the pool directory, allowing the name to be pinned explicitly.

    Defaults to ``cfg-<config_hash()>``, which is what makes an incompatible
    reuse structurally impossible rather than merely remembered.  The override
    exists for one real situation: folding the converter into the hash changed
    the name, and re-deriving it would have orphaned 1.46M already-converted
    documents whose conversions are perfectly valid.  Pinning is safe *because*
    every row now carries ``converter_id`` -- a mixed pool stays separable by
    predicate even when the directory name no longer distinguishes it.  Rows
    written before that column existed read as null, which is itself the signal.
    """
    return pool_root / (name or f"cfg-{config_hash(image_path)}")


def converter_id(image_path: str | None = None) -> str:
    """Short per-row converter fingerprint, so mixed pools stay separable."""
    identity = converter_identity(image_path)
    return (
        identity.get("image_sha256", "")[:12]
        or identity.get("latexml_version", "")
        or "unknown"
    )


def is_readable_part(path: Path) -> bool:
    """Whether a published part file is complete enough to count as done.

    Resume gated on ``exists()``, which cannot tell a finished shard from one
    whose Parquet footer was never written.  A corrupt part file was therefore
    permanent: it satisfied the skip check forever and the shard could never be
    rebuilt.  Reading the footer is cheap -- metadata only, no column data.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        pq.ParquetFile(path).metadata  # noqa: B018 - footer parse is the check
    except (OSError, pa.ArrowInvalid, pa.ArrowIOError):
        return False
    return True


def write_atomic(table: pa.Table, path: Path) -> None:
    """Publish a part file that is either complete or absent, never in between.

    The temp name carries pid and a random suffix.  Deriving it from the target
    alone meant two tasks on one shard shared a temp file and raced through
    ``os.replace``; that published unreadable Parquet in two of three trials.
    Sorting by ``tier`` before writing is what makes the row-group statistics
    selective enough for predicate pushdown to skip anything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if "tier" in table.column_names:
        table = table.sort_by([("tier", "ascending")])
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        pq.write_table(table, tmp, compression="zstd", row_group_size=ROW_GROUP_SIZE)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
