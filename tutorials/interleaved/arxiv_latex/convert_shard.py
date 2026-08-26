#!/usr/bin/env python
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

r"""Convert one whole shard, end to end, and leave nothing behind.

**The unit of work is a shard, and this is the point of the script.**  The
sampling workflow staged every project to disk first, then converted, then
packed.  Measured on the 10k sample that is 4.59 MB and ~9 loose PNGs per
document, which at 2.87M documents would be **13.2 TB of transient staging and
25.6M inodes** -- the access pattern Lustre handles worst, for data that is
deleted at the end anyway.

Here each document is extracted, converted, read into memory and *immediately*
deleted, so disk in flight is bounded by ``--procs`` documents rather than by the
corpus.  A shard coordinates with no other worker, which is what makes 12,830 of
these safe to run concurrently.

**Rows are flushed to part files incrementally, and this is what bounds RSS.**
The previous version accumulated every row of a shard and then handed the whole
list to ``Table.from_pylist``, which copies it: peak resident was a function of
the shard's HTML payload and of nothing else.  Measured worst case,
``arXiv_src_1409_008`` -- 660 documents, 1,682 MB of HTML -- reached ~3.4 GB
resident and ~6.7 GB transient, and lowering ``--procs`` from 32 to 16 could not
help because concurrency was never the axis that mattered.  Rows are now written
out every ``--flush-rows`` rows *or* every ``--flush-bytes`` of buffered HTML,
whichever comes first, as ``part-0000``, ``part-0001``, ...  The byte bound is
the one that actually covers 1409_008: 660 rows never reaches a 2,000-row
threshold, and 2,000 rows at that shard's 2.5 MB/document would be 5 GB on its
own.  Completed futures are drained inside the producer loop for the same
reason: a finished future holds its row's HTML alive just as surely as the
output buffer does.

**Assets and logs stay one-per-shard-run, deliberately.**  Both stream to disk
rather than to memory, so neither contributes to the problem flushing solves,
and rotating the asset tar mid-run would mean closing it while worker threads
hold rows destined for the previous part -- an atomicity hazard bought for no
gain.  Each is published exactly once, atomically, named for the *first* part
index this run wrote; readers join assets to documents by ``arxiv_id``, which is
inside the tar, not by part index.

**Resume is by completion marker, not by file presence.**  A shard that emits N
part files is only done once ``_meta/parts/<shard>-iter-NNN.json`` names all of
them and each one's Parquet footer parses (``pool.is_readable_part``).  Presence
alone could not distinguish a finished shard from one killed between flushes, or
from one whose footer was never written -- and a corrupt part file was therefore
permanently "done".  The marker is also the unit a later seal step folds into
``documents.parquet``, so it is not bookkeeping invented for resume alone.

Reuse is by ``source_sha256``: a shard partially converted by an earlier attempt
or an earlier iteration converts only the members the pool does not already
hold, so a killed run resumes mid-shard rather than from the top, and growing
from a sample to the corpus never redoes work.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import shutil
import sys
import tarfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TextIO

import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_curator.stages.interleaved.latex.arxiv.latexml.artifacts import scan as scan_artifacts
from nemo_curator.stages.interleaved.latex.arxiv.latexml.boilerplate import strip_boilerplate
from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import convert, error_kinds
from nemo_curator.stages.interleaved.latex.arxiv.latexml.extract import extract_submission
from nemo_curator.stages.interleaved.latex.arxiv.latexml.pool import (
    FLUSH_ROWS,
    MAX_HTML_BYTES,
    FRAME_CENSUS,
    SCHEMA,
    arxiv_id_from_dir,
    converter_id,
    is_readable_part,
    part_name,
    pool_dir,
    shard_stem,
    write_atomic,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.quality import (
    Status,
    Tier,
    assess,
    source_expects_figures,
    source_expects_math,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.sampling import parse_shard
from nemo_curator.stages.interleaved.latex.arxiv.regex.parsing import strip_comments

#: Corpus runs take every member, so no document stands for any other.
CORPUS_WEIGHT = 1.0

#: The snapshot these source tars were drawn from.  Recorded per row so a later
#: snapshot reusing this pool's conversions stays separable from this one.
DEFAULT_SNAPSHOT = "snapshot-2026-07-27"

#: Buffered HTML bytes that force a flush regardless of row count.  ``FLUSH_ROWS``
#: alone does not bound memory: the worst measured shard holds 1.68 GB of HTML in
#: 660 rows and would never reach a 2,000-row threshold at all.  Bytes are the
#: axis the OOMs were actually on, so they get their own bound.
FLUSH_BYTES = 256 * 1024 * 1024


def _part_index(path: Path) -> int:
    """``iter-003-part-0007.parquet`` -> ``7``; ``-1`` when it does not parse."""
    try:
        return int(path.stem.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def next_part_index(directory: Path, iteration: int) -> int:
    """First part index this run may write without overwriting an existing file.

    Part files are immutable (DATA_LAYOUT rule 1).  A resumed shard reuses the
    rows already flushed, so it must *append* new parts rather than restart at
    ``0000`` and overwrite a good file with a different, shorter set of rows.
    Unreadable leftovers are counted here too: they are stepped over rather than
    reused, so a killed writer costs an index, not a collision.
    """
    if not directory.exists():
        return 0
    prefix = part_name(iteration).rsplit("-", 1)[0]
    return max((_part_index(p) for p in directory.glob(f"{prefix}-*.parquet")), default=-1) + 1


def already_converted(pool: Path, shard: str) -> set[str]:
    """``source_sha256`` of every document this pool already holds for *shard*.

    Read from the pool rather than remembered in a ledger: a ledger can disagree
    with reality after a crash, and this is also what lets a corpus run inherit
    the sample's work -- and a resumed run inherit its own earlier parts --
    instead of duplicating it.
    """
    directory = pool / "html" / shard_stem(shard)
    if not directory.exists():
        return set()
    # Enumerated explicitly rather than handing the *directory* to pyarrow:
    # directory discovery reads every file it finds, so one half-written
    # ``*.parquet.tmp`` left by a killed task would raise here, be swallowed
    # below, and silently turn a resumed shard into a full re-conversion.  Each
    # candidate's footer is checked for the same reason: one unreadable part
    # must cost its own rows, not the whole shard's reuse.
    parts = sorted(str(p) for p in directory.glob("*.parquet") if is_readable_part(p))
    if not parts:
        return set()
    try:
        table = ds.dataset(parts, format="parquet").to_table(columns=["source_sha256"])
    except (OSError, pa.ArrowInvalid):
        return set()
    return {h for h in table.column("source_sha256").to_pylist() if h}


def shard_is_done(marker: Path, pool: Path) -> bool:
    """Whether *marker* names a complete, still-readable set of part files.

    Both halves matter.  Without the marker, a shard killed between flushes
    looks finished because part files exist; without re-checking the parts, a
    part file corrupted after the marker was written stays "done" forever, which
    is exactly the failure ``exists()``-based resume had.
    """
    try:
        meta = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    parts = meta.get("part_files")
    if not isinstance(parts, list):
        return False
    return all(is_readable_part(pool / rel) for rel in parts)


def build_row(  # noqa: PLR0913
    arxiv_id: str,
    shard: str,
    era: str,
    iteration: int,
    digest: str,
    snapshot: str,
    conv_id: str,
    **kw,
) -> dict:
    row = {
        "arxiv_id": arxiv_id,
        # Keeps the identifier alive through readers that project to [html, url].
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "shard": shard,
        "era": era,
        "iteration": iteration,
        # A whole-shard census: every member is taken, so no row stands for any
        # other and sample_weight is 1.0 by construction.
        "frame": FRAME_CENSUS,
        "snapshot": snapshot,
        "converter_id": conv_id,
        "source_sha256": digest,
        "root_tex": None,
        "html": None,
        "status": Status.NO_SOURCE.value,
        "tier": Tier.REJECTED.value,
        "kind": None,
        "n_warning": 0, "n_error": 0, "n_fatal": 0,
        "n_math": 0, "n_alttext": 0, "n_img": 0, "n_section": 0,
        "n_artifacts": 0,
        # Null, not False: "we never read the source" and "the source has no
        # math" are different facts, and a re-tiering pass must be able to tell
        # them apart without going back to the tars.
        "source_expects_math": None,
        "source_expects_figures": None,
        "failed_gates": [],
        "duration_s": 0.0,
        "sample_weight": CORPUS_WEIGHT,
        "content_derivation": "latex_latexml",
    }
    row.update(kw)
    return row


class AssetTar:
    """Serialises asset writes and refuses to publish a stream it corrupted.

    ``TarFile.add`` is stateful shared work: it writes a header, advances the
    logical offset, then copies bytes.  A failure *between* those steps leaves
    the offset and the file position disagreeing, and every subsequent add lands
    at a bogus position.  Reproduced: one failed add followed by two good ones
    produced an archive that failed readback entirely -- and was published
    anyway, because the only publish check was ``len(members) > 0``.

    A failed add is rewound to the byte offset it started from, which keeps the
    already-written members intact.  If the rewind itself fails there is nothing
    honest left to publish, so the tar is marked poisoned and discarded.
    """

    def __init__(self, tar: tarfile.TarFile) -> None:
        self._tar = tar
        self._lock = threading.Lock()
        self._ok = True
        self.n_added = 0
        self.n_failed = 0

    def add(self, path: Path, arcname: str) -> None:
        with self._lock:
            if not self._ok:
                return
            # Nothing in here may raise into the worker thread: an asset failure
            # is an asset failure, not a lost document row.
            try:
                offset = self._tar.offset
                position = self._tar.fileobj.tell()
                members = len(self._tar.members)
                self._tar.add(path, arcname=arcname)
            except Exception:  # noqa: BLE001 - one unreadable PNG must not cost the archive
                self.n_failed += 1
                try:
                    self._tar.fileobj.seek(position)
                    self._tar.fileobj.truncate()
                    self._tar.offset = offset
                    del self._tar.members[members:]
                except Exception:  # noqa: BLE001 - unrecoverable stream; do not publish it
                    self._ok = False
            else:
                self.n_added += 1

    @property
    def publishable(self) -> bool:
        # ``members`` rather than ``getmembers()`` -- the latter tries to read the
        # archive back, which a write-mode TarFile refuses.
        return self._ok and bool(self._tar.members)


class PartWriter:
    """Buffers rows and publishes them as ``part-NNNN`` files under two bounds.

    Rows are dropped from the Python buffer before the Arrow table is written so
    the two full copies of a shard's HTML never coexist, which was the other half
    of the measured 4x payload amplification.
    """

    def __init__(self, directory: Path, iteration: int, flush_rows: int, flush_bytes: int) -> None:
        self.dir = directory
        self._iteration = iteration
        self._flush_rows = flush_rows
        self._flush_bytes = flush_bytes
        self._index = next_part_index(directory, iteration)
        self.first_index = self._index
        self._rows: list[dict] = []
        self._bytes = 0
        self.paths: list[Path] = []
        self.n_rows = 0
        self.tiers: dict[str, int] = {}
        self.cpu_s = 0.0

    def add(self, row: dict) -> None:
        self._rows.append(row)
        self._bytes += len(row["html"] or "")
        self.n_rows += 1
        self.tiers[row["tier"]] = self.tiers.get(row["tier"], 0) + 1
        self.cpu_s += row["duration_s"]
        if len(self._rows) >= self._flush_rows or self._bytes >= self._flush_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows, schema=SCHEMA)
        self._rows, self._bytes = [], 0
        path = self.dir / f"{part_name(self._iteration, self._index)}.parquet"
        write_atomic(table, path)
        del table
        self.paths.append(path)
        self._index += 1


def process_member(  # noqa: PLR0913
    payload: bytes,
    member: str,
    digest: str,
    context: dict,
    workdir: Path,
    asset_tar: AssetTar | None,
) -> tuple[dict, str | None]:
    """Extract, convert, harvest, delete -- and never raise.

    An unexpected exception here used to propagate out of ``future.result()``
    and abandon the *whole shard*: no part file, so the resume check would hand
    the same shard back on the next attempt and it would fail the same way
    forever.  One malformed submission out of 2.87M must cost one row, not one
    shard, so the failure is recorded as a row instead.
    """
    try:
        return _convert_member(payload, member, digest, context, workdir, asset_tar)
    except Exception as exc:  # noqa: BLE001 - a bad document must not abort a shard
        arxiv_id = arxiv_id_from_dir(f"{shard_stem(context['shard'])}__{Path(member).name}")
        row = build_row(arxiv_id, **context, digest=digest,
                        status=Status.ERROR.value, tier=Tier.REJECTED.value,
                        failed_gates=["worker_exception"])
        return row, f"Fatal:worker_exception: {type(exc).__name__}: {exc}"


def _convert_member(  # noqa: PLR0913
    payload: bytes,
    member: str,
    digest: str,
    context: dict,
    workdir: Path,
    asset_tar: AssetTar | None,
) -> tuple[dict, str | None]:
    """Extract, convert, harvest, delete. Returns (row, log)."""
    # The whole member name goes to pool.arxiv_id_from_dir, extension included.
    # Stripping ``.gz`` here first is what produced ``arxiv_id='1203.5560.pdf'``
    # and a URL that 404s on 100% of pdf_only rows (~210k corpus-wide): the
    # ``.pdf`` branch returned before any ``.pdf`` strip could run.  There is now
    # exactly one place that derives an identifier, and this is not it.
    name = f"{shard_stem(context['shard'])}__{Path(member).name}"
    arxiv_id = arxiv_id_from_dir(name)
    project_dir = workdir / Path(member).name.removesuffix(".gz")
    row = functools.partial(build_row, arxiv_id, **context, digest=digest)

    try:
        if member.endswith(".pdf"):
            # No LaTeX exists; counted so the denominator stays "of all
            # submissions" rather than silently becoming "of LaTeX submissions".
            # ``status`` is a Status member and ``kind`` carries the detail:
            # writing ``kind`` into ``status`` polluted the status vocabulary
            # with ``tar``/``empty``/``single_file`` on 0.48% of live rows, and
            # ``pdf_only`` was never a Status either.  assess() already grades a
            # sourceless submission as NO_SOURCE, so this agrees with it.
            return row(kind="pdf_only"), None

        project = extract_submission(payload, member, project_dir)
        if not project.root_tex:
            return row(kind=project.kind), None

        # Read before converting: an unreadable root .tex is a defect either way,
        # and finding out first saves ~40 core-seconds of pointless conversion.
        # ``errors="replace"`` is deliberate (LaTeX is routinely latin-1 in a
        # utf-8 file); the *encoding* is pinned because leaving it out made the
        # decode depend on the container's locale.
        try:
            source = strip_comments((project_dir / project.root_tex).read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            # Never "" on failure.  An empty source makes source_expects_math()
            # false, which silently disables the no_math gate -- the one gate
            # that catches wholesale math deletion -- so a read failure would
            # grade the document *better* than it is.
            return (
                row(kind=project.kind, root_tex=project.root_tex,
                    status=Status.ERROR.value, tier=Tier.REJECTED.value,
                    failed_gates=["source_unreadable"]),
                f"Fatal:source_unreadable: {type(exc).__name__}: {exc}",
            )

        result = convert(project_dir, project.root_tex, project_dir / "_out" / "index.html")
        log = result.log
        # Checked before strip_boilerplate, and before the row is ever built: a
        # runaway document must be dropped while it is still one string, not
        # after it has been copied into a row, an Arrow table and a write buffer.
        # One paper measured 1,444 MB here and drove peak RSS to 10.2 GB on its
        # own; the flush thresholds cannot catch it, because they are tested
        # after the row is appended.  The metadata is still recorded, so the
        # document is countable rather than silently missing.
        if result.html is not None and len(result.html) > MAX_HTML_BYTES:
            oversized = len(result.html)
            del result
            return (
                row(kind=project.kind, root_tex=project.root_tex,
                    status=Status.SUSPECT_OVERSIZED.value, tier=Tier.REJECTED.value,
                    failed_gates=["oversized"]),
                f"Fatal:oversized: {oversized} bytes of HTML exceeds {MAX_HTML_BYTES}\n{log}",
            )
        html = strip_boilerplate(result.html) if result.html else None

        verdict = assess(html, source, n_error=result.n_error, n_fatal=result.n_fatal,
                         n_warning=result.n_warning, timed_out=result.timed_out,
                         error_kinds=error_kinds(log))
        artifacts = scan_artifacts(html) if html else None

        if html and asset_tar is not None:
            for asset in sorted((project_dir / "_out").glob("*.png")):
                asset_tar.add(asset, f"{arxiv_id}/{asset.name}")

        return (
            row(
                kind=project.kind,
                root_tex=project.root_tex,
                html=html,
                status=verdict.status.value,
                tier=verdict.tier.value,
                n_warning=result.n_warning, n_error=result.n_error, n_fatal=result.n_fatal,
                n_math=verdict.counts.n_math, n_alttext=verdict.counts.n_alttext,
                n_img=verdict.counts.n_img, n_section=verdict.counts.n_section,
                n_artifacts=artifacts.total if artifacts else 0,
                # Graded from the same string assess() saw, so a future change to
                # a source-dependent gate is a Parquet-only operation instead of
                # a 6.3 TB re-read of source tars.
                source_expects_math=source_expects_math(source),
                source_expects_figures=source_expects_figures(source),
                failed_gates=list(verdict.failed_gates),
                duration_s=round(result.duration_s, 3),
            ),
            log,
        )
    finally:
        # Deleted here, not at the end of the shard: this is what bounds staging
        # to --procs documents instead of a whole shard.
        shutil.rmtree(project_dir, ignore_errors=True)


def _submit(pool_exec: ThreadPoolExecutor, inflight: threading.Semaphore, *args) -> Future:
    """Acquire a permit and hand its release to the future, exception-safely.

    The permit used to be acquired before ``submit`` with the release registered
    only once ``submit`` had returned, so a raising ``submit`` leaked it.  Enough
    leaks and the producer blocks on ``acquire()`` forever: no output, no error,
    no traceback -- a hung task indistinguishable from a slow one.
    """
    inflight.acquire()
    try:
        future = pool_exec.submit(*args)
    except BaseException:
        inflight.release()
        raise
    future.add_done_callback(lambda _f: inflight.release())
    return future


def run(  # noqa: PLR0913, PLR0915, C901
    shard: str,
    src: Path,
    pool_root: Path,
    workdir: Path,
    iteration: int = 2,
    procs: int = 32,
    snapshot: str = DEFAULT_SNAPSHOT,
    image: str | None = None,
    pool_name: str | None = None,
    conv_id: str | None = None,
    flush_rows: int = FLUSH_ROWS,
    flush_bytes: int = FLUSH_BYTES,
) -> None:
    started = time.monotonic()
    pool = pool_dir(pool_root, pool_name, image)
    stem = shard_stem(shard)
    marker = pool / "_meta" / "parts" / f"{stem}-iter-{iteration:03d}.json"
    if shard_is_done(marker, pool):
        print(f"{shard}: already packed, skipping", flush=True)
        return

    parsed = parse_shard(shard)
    if parsed is None:
        msg = f"not a shard name: {shard!r}"
        raise SystemExit(msg)

    # Deriving the fingerprint means sha256 over the pinned squashfs: measured
    # 16.8 s and 5 GB of reads, per shard, which across 12,830 tasks is ~60 CPU
    # hours and 64 TB off the filesystem for one constant.  The submitter can
    # compute it once and pass it in; --image remains the correct default for a
    # one-off run, where being right matters more than being cheap.
    context = {"shard": shard, "era": parsed.era, "iteration": iteration,
               "snapshot": snapshot, "conv_id": conv_id or converter_id(image)}
    seen = already_converted(pool, shard)
    workdir = workdir / stem
    workdir.mkdir(parents=True, exist_ok=True)

    writer = PartWriter(pool / "html" / stem, iteration, flush_rows, flush_bytes)
    part = part_name(iteration, writer.first_index)
    asset_path = pool / "assets" / stem / f"{part}.tar"
    log_path = pool / "logs" / stem / f"{part}.jsonl"
    for path in (asset_path, log_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    # Dot-prefixed: a tmp left behind by a kill is then invisible to pyarrow's
    # dataset discovery and to readers globbing the tree, instead of being a
    # truncated archive sitting next to the real ones.
    asset_tmp = asset_path.with_name(f".{asset_path.name}.{os.getpid()}.tmp")
    log_tmp = log_path.with_name(f".{log_path.name}.{os.getpid()}.tmp")

    # Bounds payloads held in RAM the same way rmtree bounds bytes held on disk.
    # A ThreadPoolExecutor queue is unbounded, so the reader would otherwise race
    # ahead and hold every member of the shard in memory at once.
    inflight = threading.Semaphore(procs * 2)
    skipped, n_logs = 0, 0
    assets: AssetTar | None = None

    def take(future: Future, sink: TextIO) -> None:
        nonlocal n_logs
        row, log = future.result()
        writer.add(row)
        if log:
            sink.write(json.dumps({"arxiv_id": row["arxiv_id"], "log": log}) + "\n")
            n_logs += 1

    def drain(pending: deque[Future], keep: int, sink: TextIO) -> None:
        """Harvest finished conversions, waiting only when the queue is too deep.

        Completed futures are consumed here rather than after the executor shuts
        down: a finished future holds its row -- HTML and all -- alive, so
        leaving them queued reintroduces the O(shard) residency part flushing
        exists to remove.  Logs stream to disk for the same reason.

        **Finished futures are taken out of order, and that is the point.**  This
        loop used to be ``while len(pending) > keep: pending.popleft().result()``,
        which waits on the *oldest* conversion.  The producer cannot submit while
        it waits, so a single slow document at the head of the queue stalls the
        whole shard: workers finish their work, find nothing new queued, and
        idle.  Measured in production, that held 2-3 of 32 converter processes
        busy -- about 9% of the intended concurrency -- with a 600s timeout
        setting the worst-case stall.  Draining by completion instead of by
        arrival keeps every worker fed; the blocking wait below is kept only for
        genuine backpressure, when the queue is over its bound and there is
        nothing finished to harvest.
        """
        still_running: deque[Future] = deque()
        while pending:
            future = pending.popleft()
            if future.done():
                take(future, sink)
            else:
                still_running.append(future)
        pending.extend(still_running)
        # Backpressure only: the queue is over its bound and nothing has
        # finished, so waiting on the oldest is the correct thing to do.
        while len(pending) > keep:
            take(pending.popleft(), sink)

    try:
        with (
            tarfile.open(asset_tmp, "w") as raw_tar,
            log_tmp.open("w", encoding="utf-8") as sink,
            tarfile.open(src / shard) as tf,
        ):
            assets = AssetTar(raw_tar)
            pending: deque[Future] = deque()
            with ThreadPoolExecutor(max_workers=procs) as pool_exec:
                for info in tf:
                    if not info.isfile() or not info.name.endswith((".gz", ".pdf")):
                        continue
                    handle = tf.extractfile(info)
                    if handle is None:
                        continue
                    payload = handle.read()
                    digest = hashlib.sha256(payload).hexdigest()
                    if digest in seen:
                        skipped += 1
                        continue
                    # A shard may carry the same submission twice (resubmission
                    # under a second identifier).  Reuse is keyed on content, so
                    # the second copy is already "held" once the first is queued.
                    seen.add(digest)
                    pending.append(
                        _submit(pool_exec, inflight, process_member, payload, info.name, digest,
                                context, workdir, assets)
                    )
                    del payload
                    drain(pending, procs * 2, sink)
                drain(pending, 0, sink)
        writer.flush()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    try:
        # An asset tar is only published when it holds something *and* was never
        # corrupted: at corpus scale an empty-tar-per-shard is 12,830 pointless
        # Lustre inodes, and an unreadable one is worse than none at all.
        if assets is not None and assets.publishable:
            os.replace(asset_tmp, asset_path)
        if n_logs:
            os.replace(log_tmp, log_path)
    finally:
        asset_tmp.unlink(missing_ok=True)
        log_tmp.unlink(missing_ok=True)

    wall = time.monotonic() - started
    # Every readable part for this (shard, iteration), not only the ones *this*
    # attempt wrote.  A run that resumes after a kill reuses the earlier parts by
    # source_sha256 and may add none of its own; listing only its own output
    # would publish a marker naming zero files while the shard's rows sit in
    # part-0000 -- a shard that reads as complete and contributes nothing at seal
    # time.  Unreadable leftovers are excluded, which is what keeps the list a
    # promise rather than a glob.
    parts = sorted(
        p for p in writer.dir.glob(f"{part_name(iteration).rsplit('-', 1)[0]}-*.parquet")
        if is_readable_part(p)
    )
    # Written last, and only once every part file is on disk: this marker is the
    # resume signal, so a partially flushed shard must not carry one.  It is also
    # what a later seal step folds into documents.parquet, which is why it
    # records the row counts rather than only the file names.
    _write_marker(marker, {
        "shard": shard,
        "iteration": iteration,
        "frame": FRAME_CENSUS,
        "snapshot": snapshot,
        "converter_id": context["conv_id"],
        "pool": pool.name,
        "n_rows": writer.n_rows,
        "n_reused": skipped,
        "n_assets": assets.n_added if assets else 0,
        "n_asset_failures": assets.n_failed if assets else 0,
        "asset_tar_published": bool(assets and assets.publishable),
        "tiers": dict(sorted(writer.tiers.items())),
        "wall_s": round(wall, 1),
        "convert_cpu_s": round(writer.cpu_s, 1),
        "part_files": [str(p.relative_to(pool)) for p in parts],
        "n_parts_written": len(writer.paths),
        "log_file": str(log_path.relative_to(pool)) if n_logs else None,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    # Wall and CPU seconds are printed per shard because they are the only
    # measurement that lets the corpus projection be checked against reality
    # while the run is in flight, from the task logs alone.
    print(
        f"{shard}: {writer.n_rows} converted, {skipped} reused, "
        f"{len(writer.paths)}/{len(parts)} parts written/total, "
        f"{assets.n_added if assets else 0} assets, "
        f"tiers={dict(sorted(writer.tiers.items()))}, wall={wall:.1f}s, "
        f"convert_cpu={writer.cpu_s:.1f}s, procs={procs}",
        flush=True,
    )


def _write_marker(marker: Path, meta: dict) -> None:
    """Publish the completion marker atomically, or not at all."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, marker)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--procs", type=int, default=32)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT,
                        help="source snapshot recorded per row")
    parser.add_argument("--image", default=None,
                        help="converter squashfs, hashed into converter_id and the pool name")
    parser.add_argument("--pool-name", default=None,
                        help="pin the pool directory instead of deriving cfg-<config_hash>")
    parser.add_argument("--conv-id", default=None,
                        help="precomputed converter_id; avoids re-hashing --image once per shard")
    parser.add_argument("--flush-rows", type=int, default=FLUSH_ROWS,
                        help="rows buffered before a part file is written")
    parser.add_argument("--flush-bytes", type=int, default=FLUSH_BYTES,
                        help="buffered HTML bytes that force a part file to be written")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
