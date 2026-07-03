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

"""Generate a Nemotron-Parse JSONL manifest with byte offsets.

Reads a JSONL file where each line contains base64-encoded PDF content and
writes a companion manifest suitable for ``main.py --jsonl-base-dir``.

Each manifest line includes ``file_name``, ``jsonl_file``, and ``byte_offset``
so the preprocess stage can seek directly to the source record (O(1) per PDF).

Usage::

    python scripts/generate_jsonl_manifest.py -i data.jsonl -o /path/to/jsonl_dir

    python tutorials/interleaved/nemotron_parse_pdf/main.py \\
        --manifest /path/to/jsonl_dir/manifest.jsonl \\
        --jsonl-base-dir /path/to/jsonl_dir \\
        --output-dir /path/to/parquet_output
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
        description="Generate a byte-offset manifest for JSONL-encoded PDF datasets",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        help="Path to the source JSONL file (basename or path under --output-dir)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory for the manifest and jsonl_base_dir passed to main.py",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Manifest file path (default: <output-dir>/manifest.jsonl)",
    )
    parser.add_argument(
        "--file-name-field",
        default="file_name",
        help="JSONL field holding the PDF filename (default: file_name)",
    )
    parser.add_argument(
        "--url-field",
        default="full_name",
        help="JSONL field copied to manifest url; prefixed with --url-prefix when set (default: full_name)",
    )
    parser.add_argument(
        "--url-prefix",
        default="https://github.com/",
        help="Prefix for url when --url-field is present (default: https://github.com/)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Cap the number of manifest lines written (for testing)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Regenerate the manifest even if it already exists (default: skip existing)",
    )
    return parser


def resolve_input_file(output_dir: Path, input_file: str) -> Path:
    """Resolve --input-file to an absolute path under or relative to output_dir."""
    path = Path(input_file)
    if not path.is_absolute():
        path = output_dir / path
    return path.resolve()


def generate_manifest(
    jsonl_path: Path,
    jsonl_name: str,
    manifest_path: Path,
    *,
    file_name_field: str = "file_name",
    url_field: str = "full_name",
    url_prefix: str = "https://github.com/",
    max_entries: int | None = None,
) -> int:
    """Write manifest lines with byte offsets for each record in the source JSONL.

    ``jsonl_name`` is the basename stored in each manifest entry's ``jsonl_file``
    field so ``main.py --jsonl-base-dir`` can locate the source records.

    Returns the number of manifest entries written.
    """
    if not jsonl_path.is_file():
        msg = f"Source JSONL not found: {jsonl_path}"
        raise FileNotFoundError(msg)

    written = 0
    skipped = 0
    seen: set[str] = set()

    with jsonl_path.open("rb") as f_in, manifest_path.open("w", encoding="utf-8") as f_out:
        while True:
            if max_entries is not None and written >= max_entries:
                break

            offset = f_in.tell()
            line = f_in.readline()
            if not line:
                break

            stripped = line.strip()
            if not stripped:
                skipped += 1
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON at offset {offset}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            file_name = record.get(file_name_field)
            if not file_name:
                print(
                    f"Warning: skipping offset {offset}: missing '{file_name_field}'",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            if file_name in seen:
                print(
                    f"Warning: skipping offset {offset}: duplicate file_name '{file_name}'",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            seen.add(file_name)

            entry: dict[str, str | int] = {
                "file_name": file_name,
                "jsonl_file": jsonl_name,
                "byte_offset": offset,
            }

            url_value = record.get(url_field)
            if url_value:
                entry["url"] = f"{url_prefix}{url_value}"

            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    if skipped:
        print(f"Skipped {skipped} line(s)", file=sys.stderr)

    return written


def main() -> None:
    parser = create_argparser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.is_dir():
        print(f"Error: --output-dir is not a directory: {output_dir}", file=sys.stderr)
        sys.exit(1)

    jsonl_path = resolve_input_file(output_dir, args.input_file)
    if output_dir not in jsonl_path.parents and jsonl_path.parent != output_dir:
        print(
            f"Warning: input file {jsonl_path} is not under --output-dir {output_dir}; "
            "main.py --jsonl-base-dir must point to the directory containing the JSONL",
            file=sys.stderr,
        )

    manifest_path = (
        Path(args.manifest_output).resolve() if args.manifest_output else output_dir / "manifest.jsonl"
    )

    if manifest_path.exists() and not args.force:
        print(
            f"Manifest already exists: {manifest_path}; skipping (use --force/-f to regenerate)"
        )
        return

    # Write to a temporary <manifest>.wip and atomically rename to the final path
    # only after a complete, non-empty manifest is produced, so an interrupted run
    # never leaves a partial manifest that looks finished.
    wip_path = manifest_path.with_name(manifest_path.name + ".wip")
    if wip_path.exists():
        wip_path.unlink()

    try:
        count = generate_manifest(
            jsonl_path,
            jsonl_path.name,
            wip_path,
            file_name_field=args.file_name_field,
            url_field=args.url_field,
            url_prefix=args.url_prefix,
            max_entries=args.max_entries,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        wip_path.unlink(missing_ok=True)
        sys.exit(1)
    except BaseException:
        # Never leave a partial .wip behind on error or interrupt (e.g. Ctrl-C).
        wip_path.unlink(missing_ok=True)
        raise

    if count == 0:
        print(
            f"Error: no manifest entries written from {jsonl_path}; "
            f"check --file-name-field (got '{args.file_name_field}') and the input file",
            file=sys.stderr,
        )
        wip_path.unlink(missing_ok=True)
        sys.exit(1)

    os.replace(wip_path, manifest_path)
    print(f"Wrote {count} manifest entries to {manifest_path}")


if __name__ == "__main__":
    main()
