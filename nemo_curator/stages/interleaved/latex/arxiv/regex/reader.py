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

"""Turn arXiv LaTeX submissions into row-wise interleaved records."""

from __future__ import annotations

import gzip
import io
import json
import posixpath
import re
import tarfile
from dataclasses import dataclass
from typing import Any, Literal

import fsspec
import pyarrow as pa
from loguru import logger

from nemo_curator.core.utils import split_table_by_group
from nemo_curator.stages.interleaved.io.readers.base import BaseInterleavedReader
from nemo_curator.stages.interleaved.latex.arxiv.regex.parsing import (
    Figure,
    ParsedProject,
    TextSegment,
    guess_content_type,
    parse_project,
)
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

#: Columns this reader adds on top of ``INTERLEAVED_SCHEMA``.
EXTRA_COLUMNS: dict[str, pa.DataType] = {
    "arxiv_id": pa.string(),
    "element_class": pa.string(),
    "graphics_file": pa.string(),
    "figure_id": pa.int64(),
    "figure_label": pa.string(),
}

#: ``text_content`` MIME type for body text.  The text is still LaTeX after
#: cleaning -- detexing is intentionally left to downstream stages.
LATEX_CONTENT_TYPE = "text/x-tex"

DEFAULT_MAX_PROJECT_BYTES = 256 * 1024 * 1024

# Pre-2007 arXiv ids are ``<archive><YYMM><NNN>``; later ones are ``YYMM.NNNNN``.
# Mirrors ArxivIterator._format_arxiv_id in the text-only arXiv pipeline
# (nemo_curator/stages/text/download/arxiv/iterator.py).
_ARXIV_ID_RE = re.compile(r"^([a-zA-Z-]*)([\d.]+)$")


def format_arxiv_id(raw_id: str) -> str:
    """Normalize a raw arXiv id to its canonical form (``astro-ph0001001`` -> ``astro-ph/0001001``)."""
    match = _ARXIV_ID_RE.match(raw_id)
    if match is None:
        return raw_id
    archive, number = match.group(1), match.group(2)
    return f"{archive}/{number}" if archive else number


@dataclass
class ArxivLatexReaderStage(BaseInterleavedReader):
    r"""Read arXiv LaTeX submissions into an :class:`InterleavedBatch`.

    Input tasks come from :class:`ArxivTarPartitioningStage` and carry JSON
    entries describing a byte range inside an ``arXiv_src_*.tar`` shard.  For
    each submission this stage seeks to that range, decompresses the project,
    parses the LaTeX (see :mod:`nemo_curator.stages.interleaved.latex.arxiv.regex.parsing`),
    and emits one row per document element in reading order.

    Rows per submission
    -------------------
    ==================  ==========  =========================================
    ``modality``        ``position``  Content
    ==================  ==========  =========================================
    ``metadata``        ``-1``      JSON provenance blob in ``text_content``
    ``text``            ``0..n``    A run of body text between two figures
    ``image``           ``0..n``    Figure bytes in ``binary_content``
    ``text``            ``0..n``    The figure's caption (``element_class="caption"``)
    ==================  ==========  =========================================

    Figure bytes are materialized here rather than lazily: an image lives inside
    an inner tar, inside a gzip member, inside the shard, and that nesting is
    not addressable by the ``source_ref`` locator the materialization helpers
    understand.  ``source_ref`` therefore points at the *submission* -- shard
    path, member name, and byte range -- which is both resolvable and the right
    granularity for provenance and deduplication.

    Parameters
    ----------
    include_images
        Emit image rows.  Set ``False`` for a cheap text-only pass.
    emit_captions
        Emit each figure caption as its own text row.  Captions are emitted once
        per float, not once per sub-panel.
    clean
        Apply light LaTeX cleaning to text segments (drops in-body macro
        definitions and layout-only commands).  Meaningful markup is preserved.
    drop_bibliography, drop_appendix
        Truncate the document body at the bibliography / ``\appendix``.
    min_text_chars
        Discard text segments shorter than this after cleaning.
    max_image_bytes
        Skip figures larger than this many bytes.  ``None`` keeps every figure.
    image_content_types
        If set, keep only figures whose sniffed MIME type is in this tuple --
        e.g. ``("image/png", "image/jpeg")`` to drop PostScript figures that
        would need rasterizing.
    max_project_bytes
        Refuse to decompress a submission larger than this (decompression-bomb
        guard).
    on_missing_graphics
        ``"skip"`` drops figures whose file could not be found in the project
        (~0.2% of references).  ``"annotate"`` emits the row with
        ``materialize_error`` set instead, which the default writer policy turns
        into a hard error.
    max_batch_bytes
        Split the output into batches of roughly this size, never splitting a
        submission across batches.
    """

    include_images: bool = True
    emit_captions: bool = True
    clean: bool = True
    drop_bibliography: bool = True
    drop_appendix: bool = False
    min_text_chars: int = 1
    max_image_bytes: int | None = None
    image_content_types: tuple[str, ...] | None = None
    max_project_bytes: int = DEFAULT_MAX_PROJECT_BYTES
    on_missing_graphics: Literal["skip", "annotate"] = "skip"
    max_batch_bytes: int | None = None
    name: str = "arxiv_latex_reader"

    def __post_init__(self) -> None:
        super().__post_init__()
        self._storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

    def outputs(self) -> tuple[list[str], list[str]]:
        top_level, fields = super().outputs()
        return top_level, [*fields, "arxiv_id", "element_class"]

    # -- project loading --

    def _decompress(self, payload: bytes, member: str) -> bytes | None:
        """Gunzip *payload*, refusing anything larger than ``max_project_bytes``."""
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(payload)) as gz:
                raw = gz.read(self.max_project_bytes + 1)
        except (OSError, EOFError) as exc:
            logger.debug("{}: not a gzip stream ({})", member, exc)
            return None
        if len(raw) > self.max_project_bytes:
            logger.warning("{}: decompressed project exceeds max_project_bytes, skipping", member)
            return None
        return raw

    def _project_members(self, raw: bytes, member: str) -> dict[str, bytes]:
        """Expand a submission into ``{name: payload}``.

        A submission is either a tar of the project's files or, for ~6% of
        papers, a single bare LaTeX file.
        """
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tf:
                members: dict[str, bytes] = {}
                total = 0
                for info in tf.getmembers():
                    if not info.isfile():
                        continue
                    total += info.size
                    if total > self.max_project_bytes:
                        logger.warning("{}: project exceeds max_project_bytes while extracting, truncating", member)
                        break
                    extracted = tf.extractfile(info)
                    if extracted is not None:
                        members[info.name] = extracted.read()
                return members
        except tarfile.ReadError:
            # Single-file submission: a bare .tex (or occasionally a PDF).
            if raw.startswith(b"%PDF"):
                return {}
            stem = posixpath.basename(member).removesuffix(".gz")
            return {f"{stem}.tex": raw}
        except (OSError, EOFError) as exc:
            logger.warning("{}: could not expand submission ({})", member, exc)
            return {}

    # -- row construction --

    @staticmethod
    def _base_row(sample_id: str, arxiv_id: str, source_ref: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "position": None,
            "modality": None,
            "content_type": None,
            "text_content": None,
            "binary_content": None,
            "source_ref": source_ref,
            "materialize_error": None,
            "arxiv_id": arxiv_id,
            "element_class": None,
            "graphics_file": None,
            "figure_id": None,
            "figure_label": None,
        }

    def _keep_figure(self, payload: bytes, content_type: str) -> bool:
        if self.max_image_bytes is not None and len(payload) > self.max_image_bytes:
            return False
        return not (self.image_content_types is not None and content_type not in self.image_content_types)

    def _rows_for_submission(
        self,
        entry: dict[str, Any],
        members: dict[str, bytes],
        parsed: ParsedProject,
    ) -> list[dict[str, Any]]:
        arxiv_id = format_arxiv_id(posixpath.basename(entry["member"]).removesuffix(".gz"))
        sample_id = arxiv_id
        source_ref = InterleavedBatch.build_source_ref(
            path=entry["tar"],
            member=entry["member"],
            byte_offset=entry.get("offset"),
            byte_size=entry.get("size"),
        )

        elements = parsed.elements
        rows: list[dict[str, Any]] = []
        position = 0
        figure_id = 0
        skipped: list[str] = list(parsed.unresolved_references)

        for index, element in enumerate(elements):
            if isinstance(element, TextSegment):
                row = self._base_row(sample_id, arxiv_id, source_ref)
                row.update(
                    position=position,
                    modality="text",
                    content_type=LATEX_CONTENT_TYPE,
                    text_content=element.text,
                    element_class="text",
                )
                rows.append(row)
                position += 1
                continue

            payload = members.get(element.member) if element.member else None
            content_type = guess_content_type(element.member, payload) if element.member else None
            emitted = False
            if self.include_images and payload is not None and self._keep_figure(payload, content_type):
                row = self._base_row(sample_id, arxiv_id, source_ref)
                row.update(
                    position=position,
                    modality="image",
                    content_type=content_type,
                    binary_content=payload,
                    element_class="figure",
                    graphics_file=element.member,
                    figure_id=figure_id,
                    figure_label=element.label,
                )
                rows.append(row)
                position += 1
                emitted = True
            elif self.include_images and payload is None and self.on_missing_graphics == "annotate":
                row = self._base_row(sample_id, arxiv_id, source_ref)
                row.update(
                    position=position,
                    modality="image",
                    content_type=content_type,
                    materialize_error=f"unresolved graphics reference '{element.reference}'",
                    element_class="figure",
                    graphics_file=element.member,
                    figure_id=figure_id,
                    figure_label=element.label,
                )
                rows.append(row)
                position += 1
                emitted = True
            elif payload is None:
                skipped.append(element.reference)

            # Emit the caption once per float, after its last sub-panel, so a
            # multi-panel figure does not repeat the same text N times.
            following = elements[index + 1] if index + 1 < len(elements) else None
            last_of_float = not (isinstance(following, Figure) and following.group_index == element.group_index)
            if self.emit_captions and element.caption and last_of_float:
                row = self._base_row(sample_id, arxiv_id, source_ref)
                row.update(
                    position=position,
                    modality="text",
                    content_type=LATEX_CONTENT_TYPE,
                    text_content=element.caption,
                    element_class="caption",
                    figure_id=figure_id if emitted else None,
                    figure_label=element.label,
                )
                rows.append(row)
                position += 1
            if emitted and last_of_float:
                figure_id += 1

        if not rows:
            return []

        metadata_row = self._base_row(sample_id, arxiv_id, source_ref)
        metadata_row.update(
            position=-1,
            modality="metadata",
            content_type="application/json",
            element_class="metadata",
            text_content=json.dumps(
                {
                    "arxiv_id": arxiv_id,
                    "source_tar": entry["tar"],
                    "source_member": entry["member"],
                    "root_tex": parsed.root_tex,
                    "num_text_rows": sum(row["modality"] == "text" for row in rows),
                    "num_image_rows": sum(row["modality"] == "image" for row in rows),
                    "num_unresolved_graphics": len(skipped),
                    "graphics_paths": list(parsed.graphics_paths),
                },
                ensure_ascii=False,
            ),
        )
        return [metadata_row, *rows]

    # -- main entry point --

    def _empty_output_schema(self) -> pa.Schema:
        base = self.schema if self.schema is not None else INTERLEAVED_SCHEMA
        existing = set(base.names)
        extra = [pa.field(name, dtype) for name, dtype in EXTRA_COLUMNS.items() if name not in existing]
        return pa.schema([*base, *extra]) if extra else base

    def _read_submissions(self, task: FileGroupTask) -> list[dict[str, Any]]:
        """Read every submission referenced by *task*, one file handle per shard."""
        by_tar: dict[str, list[dict[str, Any]]] = {}
        for raw_entry in task.data:
            entry = json.loads(raw_entry)
            by_tar.setdefault(entry["tar"], []).append(entry)

        rows: list[dict[str, Any]] = []
        for tar_path, entries in by_tar.items():
            # fsspec.open() is lazy -- it returns an OpenFile without touching the
            # filesystem -- so an unreadable shard raises on __enter__, not here.
            try:
                with fsspec.open(tar_path, mode="rb", **self._storage_options) as fobj:
                    rows.extend(self._rows_for_shard(tar_path, entries, fobj))
            except Exception as exc:  # noqa: BLE001 -- an unreadable shard must not fail the task
                logger.warning("Cannot read {}: {}", tar_path, exc)
        return rows

    def _rows_for_shard(self, tar_path: str, entries: list[dict[str, Any]], fobj: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        rows: list[dict[str, Any]] = []
        for entry in entries:
            try:
                fobj.seek(entry["offset"])
                payload = fobj.read(entry["size"])
                if len(payload) != entry["size"]:
                    # Short read: remote filesystems may return fewer bytes than asked.
                    payload += fobj.read(entry["size"] - len(payload))
                rows.extend(self._rows_for_entry(entry, payload))
            except Exception as exc:  # noqa: BLE001 -- one bad paper must not fail the shard
                logger.warning("{}::{}: {}", tar_path, entry.get("member"), exc)
        return rows

    def _rows_for_entry(self, entry: dict[str, Any], payload: bytes) -> list[dict[str, Any]]:
        raw = self._decompress(payload, entry["member"])
        if raw is None:
            return []
        members = self._project_members(raw, entry["member"])
        if not members:
            return []
        parsed = parse_project(
            members,
            drop_bibliography=self.drop_bibliography,
            drop_appendix=self.drop_appendix,
            clean=self.clean,
            min_text_chars=self.min_text_chars,
        )
        if parsed.error is not None:
            logger.debug("{}: {}", entry["member"], parsed.error)
            return []
        return self._rows_for_submission(entry, members, parsed)

    def _align_output(self, table: pa.Table) -> pa.Table:
        """Reconcile reserved columns, then pin the extra columns to declared types.

        ``reconcile_schema`` only canonicalizes reserved columns; passthrough
        columns keep whatever ``Table.from_pylist`` inferred.  A task whose papers
        happen to have no figures infers ``null`` for ``figure_id`` and
        ``graphics_file`` while a task with figures infers ``int64``/``string``,
        and writing both into one Parquet dataset then fails on the schema
        mismatch.  Pinning the types here keeps every batch mutually compatible.
        """
        table = super()._align_output(table)
        if self.schema is not None:
            return table  # an explicit schema already governs every column
        for name, dtype in EXTRA_COLUMNS.items():
            index = table.schema.get_field_index(name)
            if index >= 0 and table.schema.field(index).type != dtype:
                table = table.set_column(index, pa.field(name, dtype), table.column(index).cast(dtype))
        return table

    def process(self, task: FileGroupTask) -> InterleavedBatch | list[InterleavedBatch]:
        rows = self._read_submissions(task)

        if rows:
            table = self._align_output(pa.Table.from_pylist(rows))
        else:
            table = pa.Table.from_pylist([], schema=self._empty_output_schema())
        table = self._apply_ids(task.data, table)

        # Rows for one paper are appended consecutively by _read_submissions, which
        # is what this helper requires -- it splits between groups without
        # reordering, so a paper is never divided across two batches.
        splits = split_table_by_group(table, "sample_id", max_batch_bytes=self.max_batch_bytes)
        batches: list[InterleavedBatch] = []
        for index, split in enumerate(splits):
            metadata = dict(task._metadata)
            if len(splits) > 1:
                sources = task._metadata.get("source_files") or list(task.data)
                metadata["source_files"] = [f"{path}::split_{index:05d}" for path in sources]
            if self._storage_options:
                metadata["source_storage_options"] = self._storage_options
            batches.append(
                InterleavedBatch(
                    dataset_name=task.dataset_name,
                    data=split,
                    _metadata=metadata,
                    _stage_perf=task._stage_perf,
                )
            )
        return batches if len(batches) > 1 else batches[0]
