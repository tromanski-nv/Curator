# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Generate Nemotron-Parse manifest(s) for a nested directory tree of PDFs.

Walks ``--pdf-dir`` recursively and writes one JSONL manifest line per PDF::

    {"file_name": "<path relative to --pdf-dir>"}

``file_name`` is the path *relative to* ``--pdf-dir`` because the preprocess
stage reads each PDF from ``os.path.join(pdf_dir, file_name)`` and derives
``sample_id = file_name.rsplit(".", 1)[0]``. Keeping the relative sub-path (not
just the basename) avoids collisions between identically named files in
different sub-directories and keeps resume (``--skip-existing``) correct.

Usage::

    # Single manifest for a quick salloc run
    python scripts/generate_pdf_manifest.py \\
        --pdf-dir /path/to/pdf_root -o /path/to/work/manifests

    # Sharded manifests for the chained multi-node submitter
    python scripts/generate_pdf_manifest.py \\
        --pdf-dir /path/to/pdf_root -o /path/to/work/manifests --shard-size 2000

Then run the pipeline with ``--pdf-dir /path/to/pdf_root``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def create_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate manifest(s) of relative PDF paths for a nested directory tree",
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Root directory of the (possibly nested) PDF tree. Passed to main.py as --pdf-dir.",
    )
    parser.add_argument(
        "-o",
        "--manifest-dir",
        required=True,
        help="Directory to write manifest(s) into (created if missing). Keep this OUT of a "
        "read-only source tree; e.g. under your output/work dir.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        help="PDFs per shard manifest. When set, writes part-NNNNN.manifest.jsonl files "
        "(consumed by submit_nemotron_parse_pdf_chain.sh). Default: one manifest.jsonl.",
    )
    parser.add_argument(
        "--extensions",
        default=".pdf",
        help="Comma-separated file extensions to include, case-insensitive (default: .pdf)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Cap the total number of PDFs written (for testing)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing manifest(s) (default: error out if any already exist)",
    )
    return parser


def iter_pdf_relpaths(pdf_dir: Path, extensions: tuple[str, ...], max_entries: int | None) -> list[str]:
    """Return sorted relative paths of all matching files under *pdf_dir*."""
    rels: list[str] = []
    for dirpath, dirnames, filenames in os.walk(pdf_dir):
        dirnames.sort()  # deterministic traversal
        for fname in sorted(filenames):
            if fname.lower().endswith(extensions):
                abs_path = Path(dirpath) / fname
                rels.append(os.path.relpath(abs_path, pdf_dir))
                if max_entries is not None and len(rels) >= max_entries:
                    return rels
    return rels


def _write_manifest(path: Path, file_names: list[str]) -> None:
    """Atomically write a manifest of ``{"file_name": ...}`` lines to *path*."""
    wip = path.with_name(path.name + ".wip")
    try:
        with wip.open("w", encoding="utf-8") as f:
            for name in file_names:
                f.write(json.dumps({"file_name": name}, ensure_ascii=False) + "\n")
        os.replace(wip, path)
    except BaseException:
        wip.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir).resolve()
    if not pdf_dir.is_dir():
        print(f"Error: --pdf-dir is not a directory: {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    manifest_dir = Path(args.manifest_dir).resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    extensions = tuple(e.strip().lower() for e in args.extensions.split(",") if e.strip())
    if not extensions:
        print("Error: --extensions resolved to an empty list", file=sys.stderr)
        sys.exit(1)

    rels = iter_pdf_relpaths(pdf_dir, extensions, args.max_entries)
    if not rels:
        print(f"Error: no files matching {extensions} found under {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    if args.shard_size is None:
        targets = [(manifest_dir / "manifest.jsonl", rels)]
    else:
        if args.shard_size <= 0:
            print("Error: --shard-size must be positive", file=sys.stderr)
            sys.exit(1)
        targets = [
            (manifest_dir / f"part-{i // args.shard_size:05d}.manifest.jsonl", rels[i : i + args.shard_size])
            for i in range(0, len(rels), args.shard_size)
        ]

    existing = [str(p) for p, _ in targets if p.exists()]
    if existing and not args.force:
        print(
            "Error: manifest(s) already exist (use --force to overwrite):\n  " + "\n  ".join(existing),
            file=sys.stderr,
        )
        sys.exit(1)

    for path, names in targets:
        _write_manifest(path, names)

    print(f"Wrote {len(rels)} PDF entries across {len(targets)} manifest(s) in {manifest_dir}")
    if args.shard_size is not None:
        print(f"Run the chain with:  DATASET={pdf_dir} MANIFEST_DIR={manifest_dir} SOURCE_MODE=pdf ...")


if __name__ == "__main__":
    main()
