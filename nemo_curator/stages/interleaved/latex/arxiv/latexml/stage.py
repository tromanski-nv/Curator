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

r"""Curator stages wrapping the LaTeXML converter.

This module is the only part of the ``latexml`` package that knows about
Curator.  It owns task types, scratch lifecycle and the output schema;
:mod:`..latexml.document` owns what a conversion *is*.

    tar paths  --LatexmlTarPartitioningStage-->  submission descriptors
               --LatexmlConvertStage---------->  DocumentBatch (one row per paper)
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.dataset as ds
from loguru import logger

from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import AR5IV_CONFIG, LatexmlConfig
from nemo_curator.stages.interleaved.latex.arxiv.latexml.document import (
    convert_submission,
    shard_stem,
)
from nemo_curator.stages.interleaved.latex.arxiv.latexml.model import MAX_HTML_BYTES, ConvertedDocument
from nemo_curator.stages.interleaved.utils import resolve_storage_options
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, EmptyTask, FileGroupTask

#: latexmlc is a Perl binary, not a Python package, so unlike every other
#: optional dependency in Curator it cannot be gated behind a pip extra.  The
#: check is therefore for an executable on PATH rather than an importable
#: module, and the hint points at a container rather than at ``pip install``.
LATEXMLC_INSTALL_HINT = (
    "latexmlc was not found on PATH. It is required for LaTeXML conversion and is a Perl "
    "binary, so it cannot be installed with pip. Obtain it from the public ar5iv image, "
    "which bundles LaTeXML with the ar5iv bindings this stage expects:\n"
    "    docker pull latexml/ar5ivist:2512.17    # amd64 only; no aarch64 build exists\n"
    "Then run the pipeline inside that image, or set LatexmlConvertStage(config=...) with "
    "an `executable` pointing at your own latexmlc."
)

#: One row per submission.  Deliberately flat: a downstream reader that projects
#: to ``[html, url]`` still gets an identifier, and every grading input is
#: retained so a change to a source-dependent gate can be re-evaluated from
#: Parquet instead of by re-reading terabytes of source tars.
HTML_SCHEMA = pa.schema(
    [
        ("arxiv_id", pa.string()),
        ("url", pa.string()),
        ("shard", pa.string()),
        ("snapshot", pa.string()),
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
        ("n_assets", pa.int32()),
        # Null, not False: "we never read the source" and "the source has no
        # math" are different facts, and a re-tiering pass must tell them apart.
        ("source_expects_math", pa.bool_()),
        ("source_expects_figures", pa.bool_()),
        ("failed_gates", pa.list_(pa.string())),
        # Subprocess wall clock, always -- never converter self-reported time.
        # Mixing the two made a threshold query return different populations for
        # reasons unrelated to the documents.
        ("duration_s", pa.float64()),
        ("content_derivation", pa.string()),
    ]
)

#: Outer-tar members that are submissions. ``.pdf`` is included deliberately:
#: those submissions ship no LaTeX and cannot convert, but dropping them here
#: would silently change the denominator from "of all submissions" to "of LaTeX
#: submissions". They are converted to ``pdf_only`` rows instead.
DEFAULT_SUBMISSION_EXTENSIONS: tuple[str, ...] = (".gz", ".pdf")


def _row(doc: ConvertedDocument, shard: str, snapshot: str, converter_id: str) -> dict[str, Any]:
    """One :class:`ConvertedDocument` as a row of :data:`HTML_SCHEMA`."""
    return {
        "arxiv_id": doc.arxiv_id,
        # Keeps the identifier alive through readers that project to [html, url].
        "url": f"https://arxiv.org/abs/{doc.arxiv_id}",
        "shard": shard,
        "snapshot": snapshot,
        "converter_id": converter_id,
        "source_sha256": doc.source_sha256,
        "root_tex": doc.root_tex,
        "html": doc.html,
        "status": doc.status.value,
        "tier": doc.tier.value,
        "kind": doc.kind,
        "n_warning": doc.n_warning,
        "n_error": doc.n_error,
        "n_fatal": doc.n_fatal,
        "n_math": doc.counts.n_math,
        "n_alttext": doc.counts.n_alttext,
        "n_img": doc.counts.n_img,
        "n_section": doc.counts.n_section,
        "n_artifacts": doc.n_artifacts,
        "n_assets": doc.n_assets,
        "source_expects_math": doc.source_expects_math,
        "source_expects_figures": doc.source_expects_figures,
        "failed_gates": list(doc.failed_gates),
        "duration_s": doc.duration_s,
        "content_derivation": "latex_latexml",
    }


@dataclass
class LatexmlTarPartitioningStage(ProcessingStage[FileGroupTask, FileGroupTask]):
    r"""Index arXiv source tars and emit fixed-size groups of submissions.

    Reads only the tar's member headers -- ``tarfile`` seeks rather than reads,
    so a 227 MB shard indexes in well under a second -- and hands downstream
    workers byte ranges they can seek to directly.  Each output
    :class:`FileGroupTask` carries JSON strings in ``data``::

        {"tar": "/data/arXiv_src_0001_001.tar",
         "member": "0001/astro-ph0001001.gz",
         "offset": 1536, "size": 29539}

    Unlike the sibling ``regex`` path this keeps ``.pdf`` members; see
    :data:`DEFAULT_SUBMISSION_EXTENSIONS`.
    """

    papers_per_task: int = 100
    max_papers_per_tar: int | None = None
    submission_extensions: tuple[str, ...] = DEFAULT_SUBMISSION_EXTENSIONS
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "latexml_tar_partitioning"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def __post_init__(self) -> None:
        if self.papers_per_task < 1:
            msg = f"papers_per_task must be >= 1, got {self.papers_per_task}"
            raise ValueError(msg)
        self._storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def ray_stage_spec(self) -> dict[str, Any]:
        from nemo_curator.backends.utils import RayStageSpecKeys

        return {RayStageSpecKeys.IS_FANOUT_STAGE: True}

    def _index_tar(self, tar_path: str) -> list[str]:
        entries: list[str] = []
        with (
            fsspec.open(tar_path, mode="rb", **self._storage_options) as fobj,
            tarfile.open(fileobj=fobj, mode="r:*") as tf,
        ):
            for member in tf:
                if not member.isfile() or not member.name.endswith(self.submission_extensions):
                    continue
                entries.append(
                    json.dumps(
                        {
                            "tar": tar_path,
                            "member": member.name,
                            "offset": member.offset_data,
                            "size": member.size,
                        }
                    )
                )
                if self.max_papers_per_tar is not None and len(entries) >= self.max_papers_per_tar:
                    break
        return entries

    def process(self, task: FileGroupTask) -> list[FileGroupTask]:
        tasks: list[FileGroupTask] = []
        for tar_path in task.data:
            try:
                entries = self._index_tar(tar_path)
            except (tarfile.TarError, OSError) as exc:
                # A corrupt shard must not take down the run; the rest still process.
                logger.warning("Failed to index {}: {}", tar_path, exc)
                continue
            for start in range(0, len(entries), self.papers_per_task):
                group = entries[start : start + self.papers_per_task]
                metadata = dict(task._metadata)
                metadata["source_files"] = [f"{tar_path}::papers_{start:06d}_{start + len(group):06d}"]
                metadata["partition_index"] = start // self.papers_per_task
                if self._storage_options:
                    metadata["source_storage_options"] = self._storage_options
                tasks.append(
                    FileGroupTask(
                        dataset_name=task.dataset_name,
                        data=group,
                        _metadata=metadata,
                        _stage_perf=task._stage_perf,
                    )
                )
            logger.info("{}: indexed {} submissions", tar_path, len(entries))
        return tasks


@dataclass
class LatexmlConvertStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    r"""Convert submissions to HTML5 with presentation MathML via ``latexmlc``.

    Consumes the descriptors :class:`LatexmlTarPartitioningStage` emits and
    produces one row per submission, including the ones that produced no HTML.

    Requires the ``latexmlc`` binary on ``PATH``; see
    :data:`LATEXMLC_INSTALL_HINT`.  The check runs once per worker in
    :meth:`setup`, so a missing binary fails the run immediately rather than
    once per document.

    Parameters
    ----------
    config
        What to ask ``latexmlc`` to do.  Defaults to the ar5iv configuration.
    snapshot, converter_id
        Recorded on every row so a pool spanning several source snapshots or
        converter builds stays separable after the fact.
    resume_from
        Directory of Parquet already written by an earlier run.  Submissions
        whose ``source_sha256`` appears there are skipped.  Read from the output
        rather than from a ledger: a ledger can disagree with reality after a
        crash, and reading the output is also what lets a resumed run inherit
        its own earlier parts instead of duplicating them.
    asset_dir
        If set, rasterized figures are written here as one tar per shard.
    """

    config: LatexmlConfig = AR5IV_CONFIG
    snapshot: str = ""
    converter_id: str = ""
    resume_from: str | None = None
    asset_dir: str | None = None
    scratch_dir: str | None = None
    max_html_bytes: int = MAX_HTML_BYTES
    read_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = "latexml_convert"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def __post_init__(self) -> None:
        self._storage_options = resolve_storage_options(io_kwargs=self.read_kwargs)
        self._scratch: Path | None = None
        self._seen: set[str] | None = None

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: Any = None) -> None:  # noqa: ANN401, ARG002
        """Fail fast on a missing converter, and claim a scratch directory."""
        if shutil.which(self.config.executable) is None:
            raise RuntimeError(LATEXMLC_INSTALL_HINT)
        self._scratch = Path(tempfile.mkdtemp(prefix="latexml-", dir=self.scratch_dir))
        self._seen = self._completed_hashes()
        if self._seen:
            logger.info("Resuming: {} document(s) already converted", len(self._seen))

    def teardown(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    def _completed_hashes(self) -> set[str]:
        """``source_sha256`` of every document already written under *resume_from*."""
        if not self.resume_from:
            return set()
        root = Path(self.resume_from)
        if not root.exists():
            return set()
        # Files are enumerated explicitly rather than handing the directory to
        # pyarrow: directory discovery reads everything it finds, so one
        # half-written ``*.parquet.tmp`` from a killed task would raise here and
        # silently turn a resumed run into a full re-conversion.
        parts = sorted(str(p) for p in root.rglob("*.parquet"))
        if not parts:
            return set()
        try:
            table = ds.dataset(parts, format="parquet").to_table(columns=["source_sha256"])
        except (OSError, pa.ArrowInvalid) as exc:
            logger.warning("Could not read {} for resume, converting everything: {}", root, exc)
            return set()
        return {h for h in table.column("source_sha256").to_pylist() if h}

    def _read_member(self, entry: dict[str, Any]) -> bytes:
        with fsspec.open(entry["tar"], mode="rb", **self._storage_options) as fobj:
            fobj.seek(entry["offset"])
            return fobj.read(entry["size"])

    def process(self, task: FileGroupTask) -> DocumentBatch:
        if self._scratch is None:
            msg = "setup() must run before process(); no scratch directory is claimed"
            raise RuntimeError(msg)
        seen = self._seen if self._seen is not None else set()
        rows: list[dict[str, Any]] = []
        # Keyed by shard: a task normally holds one tar's submissions, but
        # nothing in the type says so, and a single tar handle would silently
        # file one shard's figures under another's name.
        asset_tars: dict[str, tarfile.TarFile] = {}
        suffix = task._metadata.get("partition_index", 0)
        try:
            for raw in task.data:
                entry = json.loads(raw)
                shard = Path(entry["tar"]).name
                payload = self._read_member(entry)
                digest = hashlib.sha256(payload).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)

                sink = None
                if self.asset_dir:
                    # Opened on the first asset, not on the first document: most
                    # submissions rasterize nothing, and eagerly opening would
                    # leave an empty tar per shard for the reader to sift.
                    def sink(path: Path, arcname: str, _shard: str = shard) -> None:
                        if _shard not in asset_tars:
                            Path(self.asset_dir).mkdir(parents=True, exist_ok=True)
                            target = Path(self.asset_dir) / f"{shard_stem(_shard)}-{suffix:06d}.tar"
                            asset_tars[_shard] = tarfile.open(target, "w")
                        asset_tars[_shard].add(path, arcname=arcname)

                doc = convert_submission(
                    payload,
                    entry["member"],
                    shard,
                    digest,
                    self._scratch,
                    config=self.config,
                    max_html_bytes=self.max_html_bytes,
                    asset_sink=sink,
                )
                rows.append(_row(doc, shard, self.snapshot, self.converter_id))
        finally:
            for handle in asset_tars.values():
                handle.close()

        return DocumentBatch(
            dataset_name=task.dataset_name,
            data=pa.Table.from_pylist(rows, schema=HTML_SCHEMA),
            _metadata=dict(task._metadata),
            _stage_perf=task._stage_perf,
        )


@dataclass
class ArxivLatexmlReader(CompositeStage[EmptyTask, DocumentBatch]):
    """Read arXiv source tars and convert them with LaTeXML.

    Wires the three stages the conversion needs, so a caller adds one stage
    rather than three::

        pipeline.add_stage(ArxivLatexmlReader(file_paths="/data/arxiv-src/"))
    """

    file_paths: str | list[str] = ""
    papers_per_task: int = 100
    max_papers_per_tar: int | None = None
    tars_per_group: int = 1
    config: LatexmlConfig = AR5IV_CONFIG
    snapshot: str = ""
    converter_id: str = ""
    resume_from: str | None = None
    asset_dir: str | None = None
    scratch_dir: str | None = None
    read_kwargs: dict[str, Any] = field(default_factory=dict)

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                files_per_partition=self.tars_per_group,
                file_extensions=[".tar"],
                storage_options=resolve_storage_options(io_kwargs=self.read_kwargs) or None,
            ),
            LatexmlTarPartitioningStage(
                papers_per_task=self.papers_per_task,
                max_papers_per_tar=self.max_papers_per_tar,
                read_kwargs=self.read_kwargs,
            ),
            LatexmlConvertStage(
                config=self.config,
                snapshot=self.snapshot,
                converter_id=self.converter_id,
                resume_from=self.resume_from,
                asset_dir=self.asset_dir,
                scratch_dir=self.scratch_dir,
                read_kwargs=self.read_kwargs,
            ),
        ]
