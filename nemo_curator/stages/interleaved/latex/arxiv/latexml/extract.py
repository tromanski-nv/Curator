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

r"""Materialize an arXiv submission's source tree on disk, intact.

LaTeXML compiles a *project*, not a string, so the whole source tree has to
exist as real files: ``\input`` targets, ``.sty``/``.cls`` files the author
shipped, ``.bbl`` bibliographies and every figure the document references.  This
module unpacks one submission into a directory and picks its root document.

Two rules from the conversion research are enforced here:

* Never concatenate ``.tex`` files and never regex-strip before parsing -- the
  converter needs the original text.
* Several genuine root documents means several document *candidates*, not one
  joined document.  Only the chosen root is converted, and the rest are recorded.

Extraction is hostile-input safe: arXiv tarballs are author-supplied, so member
paths are validated against traversal (``../``), absolute paths, symlinks, and
device files, and the total unpacked size is capped.
"""

from __future__ import annotations

import gzip
import io
import posixpath
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

#: Members treated as LaTeX source when locating the root document.
TEX_EXTENSIONS: tuple[str, ...] = (".tex", ".ltx", ".latex")

DEFAULT_MAX_PROJECT_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 5000

_DOCUMENT_CLASS_RE = re.compile(r"\\document(?:class|style)\b")
_BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{document\}")
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\b\s*(?:\{([^{}]*)\}|([^\s\\{}%]+))")
#: Root-document filenames that win a tie, in order of preference.
_PREFERRED_ROOT_NAMES: tuple[str, ...] = ("ms", "main", "paper", "article", "manuscript", "root")


class UnsafeMemberError(Exception):
    """A tar member was rejected as unsafe to extract."""


@dataclass
class ExtractedProject:
    """One unpacked submission."""

    directory: Path
    root_tex: str | None = None
    """Project-relative path of the document to convert."""

    other_roots: tuple[str, ...] = ()
    """Additional documents that also look like roots -- separate candidates."""

    members: tuple[str, ...] = ()
    total_bytes: int = 0
    kind: str = "tar"
    """``tar``, ``single_file``, ``pdf_only`` or ``empty``."""

    warnings: list[str] = field(default_factory=list)

    @property
    def root_path(self) -> Path | None:
        return self.directory / self.root_tex if self.root_tex else None


def _is_safe_member(member: tarfile.TarInfo) -> bool:
    """Reject traversal, absolute paths, symlinks and device/FIFO members."""
    if not (member.isfile() or member.isdir()):
        return False  # symlinks, hardlinks, devices, FIFOs
    name = member.name.replace("\\", "/")
    if name.startswith(("/", "~")):
        return False
    normalized = posixpath.normpath(name)
    return not (normalized.startswith("../") or normalized == ".." or posixpath.isabs(normalized))


def _safe_extract(
    payload: bytes,
    destination: Path,
    max_bytes: int,
    max_members: int,
) -> tuple[list[str], int, list[str]]:
    """Extract a tar payload into *destination*, skipping unsafe members."""
    names: list[str] = []
    warnings: list[str] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as tf:
        for member in tf:
            if len(names) >= max_members:
                warnings.append(f"stopped at max_members={max_members}")
                break
            if not _is_safe_member(member):
                warnings.append(f"skipped unsafe member {member.name!r}")
                continue
            if member.isdir():
                continue
            total += member.size
            if total > max_bytes:
                warnings.append(f"stopped at max_project_bytes={max_bytes}")
                break
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            target = destination / posixpath.normpath(member.name.replace("\\", "/"))
            # Belt and braces: confirm the resolved path stayed inside destination.
            try:
                target.resolve().relative_to(destination.resolve())
            except ValueError:
                warnings.append(f"skipped escaping member {member.name!r}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())
            names.append(posixpath.normpath(member.name.replace("\\", "/")))
    return names, total, warnings


def _looks_like_root(text: str) -> bool:
    return bool(_DOCUMENT_CLASS_RE.search(text) or _BEGIN_DOCUMENT_RE.search(text))


def _uncommented(payload: bytes) -> str:
    r"""Decode *payload* and strip LaTeX comments before any structural scan.

    Scanning raw bytes treats commented-out markup as live.  Both failure modes
    are silent and produce a plausible HTML document built from the wrong file:

    * ``% superseded, was: \input{main}`` marks the real root as *included*,
      removing it from the root set with nothing recorded in ``other_roots``.
    * ``%\documentclass{article}`` in a fragment promotes it to a root candidate.

    Reuses the parser's comment handling, which also protects verbatim bodies,
    rather than re-deriving it here.
    """
    from nemo_curator.stages.interleaved.latex.arxiv.latexml.source_text import decode_text, strip_comments

    return strip_comments(decode_text(payload))


def _included_names(text: str, available: set[str]) -> set[str]:
    """Names this file pulls in via ``\\input``/``\\include``/``\\subfile``."""
    found: set[str] = set()
    lowered = {name.lower(): name for name in available}
    for match in _INPUT_RE.finditer(text):
        ref = (match.group(1) or match.group(2) or "").strip().strip('"')
        if not ref or "\\" in ref or "#" in ref:
            continue
        ref = posixpath.normpath(ref.removeprefix("./"))
        for suffix in ("", *TEX_EXTENSIONS):
            hit = lowered.get(f"{ref}{suffix}".lower())
            if hit is not None:
                found.add(hit)
                break
    return found


def select_roots(directory: Path, members: list[str]) -> tuple[str | None, tuple[str, ...]]:
    r"""Choose the root document and report any other genuine root candidates.

    A root declares ``\documentclass``/``\documentstyle`` (or at least
    ``\begin{document}``) and is not pulled in by another file.  Comments are
    stripped first, so commented-out markup cannot promote or demote a file.

    Ties break on directory depth first -- a top-level document beats a sample
    ``main.tex`` inside a shipped journal template -- then on a conventional
    filename, then size, then name for determinism.
    """
    tex_members = [name for name in members if name.lower().endswith(TEX_EXTENSIONS)]
    if not tex_members:
        return None, ()

    payloads: dict[str, str] = {}
    for name in tex_members:
        try:
            payloads[name] = _uncommented((directory / name).read_bytes())
        except OSError:  # pragma: no cover - unreadable file after a successful write
            continue

    candidates = [name for name, payload in payloads.items() if _looks_like_root(payload)]
    if not candidates:
        candidates = list(payloads)
    if not candidates:
        return None, ()

    available = set(payloads)
    included: set[str] = set()
    for name, payload in payloads.items():
        included |= _included_names(payload, available) - {name}
    roots = [name for name in candidates if name not in included] or candidates

    def rank(name: str) -> tuple[int, int, int, str]:
        stem = Path(name).stem.lower()
        preference = _PREFERRED_ROOT_NAMES.index(stem) if stem in _PREFERRED_ROOT_NAMES else len(_PREFERRED_ROOT_NAMES)
        depth = name.count("/")
        return (depth, preference, -len(payloads[name]), name)

    ordered = sorted(roots, key=rank)
    return ordered[0], tuple(ordered[1:])


def extract_submission(  # noqa: PLR0911 -- each early return is a distinct, documented submission shape
    payload: bytes,
    member_name: str,
    destination: Path,
    *,
    max_bytes: int = DEFAULT_MAX_PROJECT_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> ExtractedProject:
    """Unpack one arXiv submission into *destination*.

    Args:
        payload: Raw bytes of the outer-tar member (a gzip stream, or a bare PDF).
        member_name: Member name inside the shard, e.g. ``0001/astro-ph0001001.gz``.
        destination: Directory to unpack into; created if absent.
        max_bytes: Cap on total unpacked size (decompression-bomb guard).
        max_members: Cap on number of files.

    Returns:
        An :class:`ExtractedProject`.  Failures are reported through ``kind`` and
        ``warnings`` rather than raised -- one bad submission must not stop a shard.
    """
    destination.mkdir(parents=True, exist_ok=True)
    stem = posixpath.basename(member_name).removesuffix(".gz")

    if payload[:4] == b"%PDF":
        return ExtractedProject(directory=destination, kind="pdf_only", warnings=["submission is a bare PDF"])

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as gz:
            raw = gz.read(max_bytes + 1)
    except (OSError, EOFError) as exc:
        return ExtractedProject(directory=destination, kind="empty", warnings=[f"not a gzip stream: {exc}"])

    if len(raw) > max_bytes:
        return ExtractedProject(
            directory=destination, kind="empty", warnings=[f"exceeds max_project_bytes={max_bytes}"]
        )
    if raw[:4] == b"%PDF":
        return ExtractedProject(directory=destination, kind="pdf_only", warnings=["submission is a bare PDF"])

    try:
        names, total, warnings = _safe_extract(raw, destination, max_bytes, max_members)
        kind = "tar"
    except tarfile.ReadError:
        # ~30% of submissions are a single bare .tex rather than a tarball.
        target = destination / f"{stem}.tex"
        target.write_bytes(raw)
        names, total, warnings, kind = [target.name], len(raw), [], "single_file"
    except (OSError, EOFError) as exc:
        return ExtractedProject(directory=destination, kind="empty", warnings=[f"could not expand: {exc}"])

    if not names:
        return ExtractedProject(directory=destination, kind="empty", warnings=[*warnings, "no usable members"])

    root, others = select_roots(destination, names)
    if root is None:
        warnings.append("no LaTeX source found")
    if others:
        logger.debug("{}: {} additional root candidate(s): {}", member_name, len(others), others)

    return ExtractedProject(
        directory=destination,
        root_tex=root,
        other_roots=others,
        members=tuple(names),
        total_bytes=total,
        kind=kind,
        warnings=warnings,
    )
