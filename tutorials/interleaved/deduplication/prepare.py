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

"""Prepare auditable sample, arXiv-version, and PDF-SHA deduplication artifacts."""

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ARXIV_VERSION_RE = re.compile(r"v(\d+)$")
NEW_STYLE_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
GIT_EXECUTABLE = "/usr/bin/git"
MIN_DUPLICATE_GROUP_SIZE = 2


def _parquet_files(path: str) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    # Skip sidecar/metadata parquet whose names start with "_" or "." (e.g. the
    # Nemotron Parse "_perf_stats_*.parquet" files and atomic-write ".tmp" files),
    # following the Spark/Hive convention of ignoring such files in a dataset.
    return sorted(
        file
        for file in root.rglob("*.parquet")
        if not file.name.startswith(("_", "."))
    )


def _atomic_write_parquet(df: pd.DataFrame, output_file: Path, schema: pa.Schema | None = None) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_suffix(f"{output_file.suffix}.tmp.{os.getpid()}")
    table = (
        pa.Table.from_pylist([], schema=schema)
        if schema is not None and len(df) == 0
        else pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    )
    pq.write_table(table, temporary_file)
    os.replace(temporary_file, output_file)


def _write_json(payload: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def _code_state() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(  # noqa: S603
            [GIT_EXECUTABLE, "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(  # noqa: S603
                [GIT_EXECUTABLE, "-C", str(repo_root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"code_path": str(repo_root), "git_commit": commit, "git_dirty": dirty}


def _read_parquet_dataset(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    files = _parquet_files(path)
    if not files:
        msg = f"No parquet files found at {path}"
        raise FileNotFoundError(msg)
    return ds.dataset([str(file) for file in files], format="parquet").to_table(columns=columns).to_pandas()


def _normalize_arxiv_id(value: str) -> tuple[str, int | None]:
    normalized = value.removesuffix(".pdf").removeprefix("arXiv:")
    basename = normalized.rsplit("/", 1)[-1]
    version_match = ARXIV_VERSION_RE.search(basename)
    explicit_version = int(version_match.group(1)) if version_match else None
    if version_match:
        normalized = normalized[: -len(version_match.group(0))]
        basename = normalized.rsplit("/", 1)[-1]
    if NEW_STYLE_ARXIV_ID_RE.fullmatch(basename):
        return basename, explicit_version
    return normalized, explicit_version


def build_baseline(args: argparse.Namespace) -> None:
    started = time.time()
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = Path(args.input_path)
    files = _parquet_files(args.input_path)
    if not files:
        msg = f"No parquet files found at {args.input_path}"
        raise FileNotFoundError(msg)
    total_input_rows = sum(pq.ParquetFile(input_file).metadata.num_rows for input_file in files)

    samples_written = 0
    rows_seen = 0
    resumed_partitions = 0
    for input_file in files:
        relative_path = str(input_file.relative_to(input_root)) if input_root.is_dir() else input_file.name
        partition_key = hashlib.sha256(relative_path.encode()).hexdigest()[:20]
        output_file = output_dir / f"part-{partition_key}.parquet"
        if output_file.exists() and not args.overwrite:
            resumed_partitions += 1
            continue

        table = pq.read_table(input_file, columns=[args.sample_id_field, args.pdf_name_field])
        rows_seen += table.num_rows
        frame = table.to_pandas().drop_duplicates()
        sample_counts = frame.groupby(args.sample_id_field)[args.pdf_name_field].nunique(dropna=False)
        if (sample_counts != 1).any():
            examples = sample_counts[sample_counts != 1].head().to_dict()
            msg = f"Samples map to multiple PDFs in {input_file}: {examples}"
            raise ValueError(msg)

        frame = frame.drop_duplicates(subset=[args.sample_id_field], keep="first")
        frame["source_parquet"] = str(input_file)
        frame["source_path"] = frame[args.pdf_name_field].map(lambda value: str(Path(args.pdf_root) / str(value)))
        frame["mapping_error"] = pd.NA
        expected_sample_ids = frame[args.pdf_name_field].astype(str).str.removesuffix(".pdf")
        mismatched = frame[args.sample_id_field].astype(str) != expected_sample_ids
        missing = ~frame["source_path"].map(lambda path: Path(path).is_file())
        frame.loc[mismatched, "mapping_error"] = "sample_id_pdf_name_mismatch"
        frame.loc[missing, "mapping_error"] = frame.loc[missing, "mapping_error"].fillna("missing_pdf")
        baseline_schema = pa.schema(
            [
                pa.field(args.sample_id_field, pa.string()),
                pa.field(args.pdf_name_field, pa.string()),
                pa.field("source_parquet", pa.string()),
                pa.field("source_path", pa.string()),
                pa.field("mapping_error", pa.string()),
            ],
        )
        _atomic_write_parquet(frame, output_file, baseline_schema)
        samples_written += len(frame)

    inventory = _read_parquet_dataset(args.output_path)
    duplicate_samples = inventory[args.sample_id_field].duplicated(keep=False)
    duplicate_pdfs = inventory[args.pdf_name_field].duplicated(keep=False)
    mapping_errors = inventory["mapping_error"].notna()
    manifest = {
        "stage": "baseline",
        "input_path": args.input_path,
        "pdf_root": args.pdf_root,
        "output_path": args.output_path,
        "num_input_parquet_files": len(files),
        "num_partitions_resumed": resumed_partitions,
        "num_samples_written_this_run": samples_written,
        "num_rows_seen_this_run": rows_seen,
        "num_input_rows": total_input_rows,
        "num_inventory_rows": len(inventory),
        "num_unique_samples": int(inventory[args.sample_id_field].nunique()),
        "num_duplicate_sample_rows": int(duplicate_samples.sum()),
        "num_duplicate_pdf_rows": int(duplicate_pdfs.sum()),
        "num_mapping_errors": int(mapping_errors.sum()),
        "runtime_seconds": time.time() - started,
        **_code_state(),
    }
    _write_json(manifest, args.manifest_path)
    if duplicate_samples.any() or duplicate_pdfs.any() or mapping_errors.any():
        msg = f"Baseline validation failed; inspect {args.manifest_path} and {args.output_path}"
        raise RuntimeError(msg)


def _iter_metadata_records(path: str) -> Iterable[dict[str, Any]]:
    metadata_path = Path(path)
    if metadata_path.suffix == ".parquet" or metadata_path.is_dir():
        frame = _read_parquet_dataset(path)
        yield from frame.to_dict("records")
        return
    with metadata_path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    msg = f"Invalid metadata JSON on line {line_number}: {error}"
                    raise ValueError(msg) from error


def _metadata_versions(record: dict[str, Any]) -> tuple[int | None, str | None, int, str | None]:
    versions = record.get("versions") or []
    parsed: list[tuple[int, str | None]] = []
    for version in versions:
        label = str(version.get("version", ""))
        match = re.fullmatch(r"v(\d+)", label)
        if match is None:
            return None, None, len(versions), f"malformed_version:{label}"
        parsed.append((int(match.group(1)), version.get("created")))
    if not parsed:
        return None, None, 0, "missing_versions"
    version_numbers = [version_number for version_number, _ in parsed]
    if len(version_numbers) != len(set(version_numbers)):
        return None, None, len(parsed), "duplicate_version_numbers"
    selected_number, selected_created = max(parsed, key=lambda item: item[0])
    return selected_number, selected_created, len(parsed), None


def select_arxiv_versions(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    started = time.time()
    inventory = _read_parquet_dataset(
        args.inventory_path,
        columns=[args.sample_id_field, args.pdf_name_field, "source_path", "source_parquet"],
    )
    normalized = inventory[args.sample_id_field].astype(str).map(_normalize_arxiv_id)
    inventory["arxiv_id"] = normalized.map(lambda value: value[0])
    inventory["local_explicit_version"] = normalized.map(lambda value: value[1])
    wanted_ids = set(inventory["arxiv_id"])

    metadata_by_id: dict[str, dict[str, Any]] = {}
    duplicate_metadata_ids: set[str] = set()
    for record in _iter_metadata_records(args.metadata_path):
        raw_id = str(record.get("id") or record.get("arxiv_id") or "")
        arxiv_id, _ = _normalize_arxiv_id(raw_id)
        if arxiv_id not in wanted_ids:
            continue
        if arxiv_id in metadata_by_id:
            duplicate_metadata_ids.add(arxiv_id)
        metadata_by_id[arxiv_id] = record

    output_rows: list[dict[str, Any]] = []
    exception_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    fatal_exceptions = 0
    for arxiv_id, local_group in inventory.groupby("arxiv_id", sort=True):
        record = metadata_by_id.get(arxiv_id)
        selected_version = None
        selected_created = None
        version_count = 0
        metadata_error = None
        if record is None:
            metadata_error = "missing_metadata"
        else:
            selected_version, selected_created, version_count, metadata_error = _metadata_versions(record)
        if arxiv_id in duplicate_metadata_ids:
            metadata_error = "duplicate_metadata_records"

        selected_indices: set[int]
        if len(local_group) == 1:
            selected_indices = {int(local_group.index[0])}
            status = "selected_single_snapshot_pdf" if metadata_error is None else f"selected_{metadata_error}"
        else:
            matches = local_group.index[local_group["local_explicit_version"] == selected_version].tolist()
            if len(matches) == 1:
                selected_indices = {int(matches[0])}
                status = "selected_latest_explicit_version"
            else:
                selected_indices = set()
                status = "ambiguous_local_versions"
                metadata_error = metadata_error or "no_unique_latest_local_version"
                fatal_exceptions += 1

        for index, row in local_group.iterrows():
            selected = int(index) in selected_indices
            output_row = {
                **row.to_dict(),
                "version_count": version_count,
                "selected_version": f"v{selected_version}" if selected_version is not None else None,
                "selected_version_created": selected_created,
                "metadata_update_date": record.get("update_date") if record else None,
                "selection_status": status if selected else "removed_older_version",
                "selected": selected,
            }
            output_rows.append(output_row)
            if not selected:
                removed_rows.append(
                    {
                        args.sample_id_field: row[args.sample_id_field],
                        "arxiv_id": arxiv_id,
                        "selected_version": output_row["selected_version"],
                        "selection_status": output_row["selection_status"],
                    },
                )
        if metadata_error is not None:
            exception_rows.append({"arxiv_id": arxiv_id, "error": metadata_error})

    version_schema = pa.Table.from_pylist(output_rows).schema
    removed_schema = pa.schema(
        [
            pa.field(args.sample_id_field, pa.string()),
            pa.field("arxiv_id", pa.string()),
            pa.field("selected_version", pa.string()),
            pa.field("selection_status", pa.string()),
        ],
    )
    exception_schema = pa.schema([pa.field("arxiv_id", pa.string()), pa.field("error", pa.string())])
    _atomic_write_parquet(pd.DataFrame(output_rows), Path(args.output_path) / "part-00000.parquet", version_schema)
    _atomic_write_parquet(pd.DataFrame(removed_rows), Path(args.removed_path) / "part-00000.parquet", removed_schema)
    _atomic_write_parquet(
        pd.DataFrame(exception_rows),
        Path(args.exceptions_path) / "part-00000.parquet",
        exception_schema,
    )

    manifest = {
        "stage": "arxiv_version_selection",
        "input_inventory_path": args.inventory_path,
        "metadata_path": args.metadata_path,
        "metadata_sha256": args.metadata_sha256,
        "metadata_source_url": args.metadata_source_url,
        "metadata_snapshot_date": args.metadata_snapshot_date,
        "output_path": args.output_path,
        "removed_path": args.removed_path,
        "exceptions_path": args.exceptions_path,
        "num_input_samples": len(inventory),
        "num_selected_samples": sum(row["selected"] for row in output_rows),
        "num_removed_samples": len(removed_rows),
        "num_metadata_exceptions": len(exception_rows),
        "num_fatal_exceptions": fatal_exceptions,
        "metadata_coverage": (len(wanted_ids) - sum(row["error"] == "missing_metadata" for row in exception_rows))
        / max(1, len(wanted_ids)),
        "runtime_seconds": time.time() - started,
        **_code_state(),
    }
    _write_json(manifest, args.manifest_path)
    if fatal_exceptions:
        msg = f"Version selection found {fatal_exceptions} ambiguous local version group(s)"
        raise RuntimeError(msg)


def select_sha_duplicates(args: argparse.Namespace) -> None:
    started = time.time()
    inventory = _read_parquet_dataset(args.inventory_path)
    duplicated_samples = inventory[args.sample_id_field].duplicated(keep=False)
    hash_errors = inventory["hash_error"].notna() | inventory["sha256"].isna()
    if duplicated_samples.any() or hash_errors.any():
        msg = (
            f"SHA inventory is invalid: duplicate sample rows={int(duplicated_samples.sum())}, "
            f"hash errors={int(hash_errors.sum())}"
        )
        raise RuntimeError(msg)

    duplicate_rows = []
    sorted_inventory = inventory.sort_values(args.sample_id_field)
    for _, group in sorted_inventory.groupby(["size_bytes", "sha256"], sort=True, dropna=False):
        if len(group) < MIN_DUPLICATE_GROUP_SIZE:
            continue
        keeper = group.iloc[0]
        for _, duplicate in group.iloc[1:].iterrows():
            duplicate_rows.append(
                {
                    args.sample_id_field: duplicate[args.sample_id_field],
                    "keeper_sample_id": keeper[args.sample_id_field],
                    "size_bytes": int(duplicate["size_bytes"]),
                    "sha256": duplicate["sha256"],
                    "source_path": duplicate["source_path"],
                    "keeper_source_path": keeper["source_path"],
                },
            )

    output_schema = pa.schema(
        [
            pa.field(args.sample_id_field, pa.string()),
            pa.field("keeper_sample_id", pa.string()),
            pa.field("size_bytes", pa.int64()),
            pa.field("sha256", pa.string()),
            pa.field("source_path", pa.string()),
            pa.field("keeper_source_path", pa.string()),
        ],
    )
    _atomic_write_parquet(
        pd.DataFrame(duplicate_rows),
        Path(args.output_path) / "part-00000.parquet",
        output_schema,
    )
    manifest = {
        "stage": "pdf_sha256_duplicate_selection",
        "input_inventory_path": args.inventory_path,
        "output_path": args.output_path,
        "num_input_samples": len(inventory),
        "num_duplicate_samples": len(duplicate_rows),
        "num_output_samples": len(inventory) - len(duplicate_rows),
        "keeper_policy": "lexicographically_smallest_sample_id",
        "runtime_seconds": time.time() - started,
        **_code_state(),
    }
    _write_json(manifest, args.manifest_path)


def _read_removal_counts(path_or_pattern: str) -> dict[str, int]:
    """Read removal-stage sample counts, aggregating per-shard array manifests.

    A single Ray job writes one removal manifest, but the Slurm-array removal
    writes one manifest per shard (``*.shard-<i>-of-<n>.json``). Accept either a
    single file path or a glob pattern; when multiple shard manifests match, sum
    their sample counts so accounting sees one set of stage totals.
    """
    matches = sorted(glob.glob(path_or_pattern))
    if not matches:
        msg = f"No removal manifest found matching {path_or_pattern!r}"
        raise FileNotFoundError(msg)
    totals = {"num_samples_in": 0, "num_samples_removed": 0, "num_samples_out": 0}
    for match in matches:
        payload = json.loads(Path(match).read_text())
        metadata = payload.get("metadata", payload)
        for key in totals:
            totals[key] += int(metadata.get(key, 0))
    return totals


def validate_accounting(args: argparse.Namespace) -> None:
    def read_manifest(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text())

    baseline = read_manifest(args.baseline_manifest)
    versions = (
        read_manifest(args.version_manifest)
        if args.version_manifest
        else {
            "num_input_samples": baseline["num_unique_samples"],
            "num_removed_samples": 0,
            "num_selected_samples": baseline["num_unique_samples"],
        }
    )
    sha = read_manifest(args.sha_manifest)
    exact_counts = _read_removal_counts(args.exact_removal_manifest)
    fuzzy_counts = _read_removal_counts(args.fuzzy_removal_manifest)

    counts = {
        "raw": int(baseline["num_unique_samples"]),
        "version_removed": int(versions["num_removed_samples"]),
        "after_version": int(versions["num_selected_samples"]),
        "sha_removed": int(sha["num_duplicate_samples"]),
        "after_sha": int(sha["num_output_samples"]),
        "exact_removed": int(exact_counts["num_samples_removed"]),
        "after_exact": int(exact_counts["num_samples_out"]),
        "fuzzy_removed": int(fuzzy_counts["num_samples_removed"]),
        "final": int(fuzzy_counts["num_samples_out"]),
    }
    checks = {
        "version_input_matches_baseline": int(versions["num_input_samples"]) == counts["raw"],
        "after_version": counts["after_version"] == counts["raw"] - counts["version_removed"],
        "sha_input_matches_version": int(sha["num_input_samples"]) == counts["after_version"],
        "after_sha": counts["after_sha"] == counts["after_version"] - counts["sha_removed"],
        "exact_input_matches_sha": int(exact_counts["num_samples_in"]) == counts["after_sha"],
        "after_exact": counts["after_exact"] == counts["after_sha"] - counts["exact_removed"],
        "fuzzy_input_matches_exact": int(fuzzy_counts["num_samples_in"]) == counts["after_exact"],
        "final": counts["final"] == counts["after_exact"] - counts["fuzzy_removed"],
    }
    payload = {
        "stage": "deduplication_accounting",
        "counts": counts,
        "checks": checks,
        "valid": all(checks.values()),
        "source_manifests": {
            "baseline": args.baseline_manifest,
            "versions": args.version_manifest,
            "sha": args.sha_manifest,
            "exact_removal": args.exact_removal_manifest,
            "fuzzy_removal": args.fuzzy_removal_manifest,
        },
        **_code_state(),
    }
    _write_json(payload, args.output_path)
    if not payload["valid"]:
        failed_checks = [name for name, passed in checks.items() if not passed]
        msg = f"Deduplication accounting failed: {failed_checks}"
        raise RuntimeError(msg)


def _add_common_fields(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sample-id-field", default="sample_id")
    parser.add_argument("--pdf-name-field", default="pdf_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--input-path", required=True)
    baseline.add_argument("--pdf-root", required=True)
    baseline.add_argument("--output-path", required=True)
    baseline.add_argument("--manifest-path", required=True)
    baseline.add_argument("--overwrite", action="store_true")
    _add_common_fields(baseline)
    baseline.set_defaults(func=build_baseline)

    versions = subparsers.add_parser("arxiv-versions")
    versions.add_argument("--inventory-path", required=True)
    versions.add_argument("--metadata-path", required=True)
    versions.add_argument("--metadata-sha256", required=True)
    versions.add_argument("--metadata-source-url", required=True)
    versions.add_argument("--metadata-snapshot-date", required=True)
    versions.add_argument("--output-path", required=True)
    versions.add_argument("--removed-path", required=True)
    versions.add_argument("--exceptions-path", required=True)
    versions.add_argument("--manifest-path", required=True)
    _add_common_fields(versions)
    versions.set_defaults(func=select_arxiv_versions)

    sha_select = subparsers.add_parser("sha-select")
    sha_select.add_argument("--inventory-path", required=True)
    sha_select.add_argument("--output-path", required=True)
    sha_select.add_argument("--manifest-path", required=True)
    sha_select.add_argument("--sample-id-field", default="sample_id")
    sha_select.set_defaults(func=select_sha_duplicates)

    accounting = subparsers.add_parser("accounting")
    accounting.add_argument("--baseline-manifest", required=True)
    accounting.add_argument("--version-manifest")
    accounting.add_argument("--sha-manifest", required=True)
    accounting.add_argument("--exact-removal-manifest", required=True)
    accounting.add_argument("--fuzzy-removal-manifest", required=True)
    accounting.add_argument("--output-path", required=True)
    accounting.set_defaults(func=validate_accounting)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
