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

"""Run resumable interleaved PDF-SHA, exact-text, and fuzzy deduplication stages."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_curator.backends.ray_actor_pool import RayActorPoolExecutor
from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.interleaved.deduplication.pdf_sha import PdfSha256InventoryStage
from nemo_curator.stages.interleaved.deduplication.removal import InterleavedSampleIdRemovalStage
from nemo_curator.stages.interleaved.deduplication.removal_workflow import (
    InterleavedDuplicatesRemovalWorkflow,
)
from nemo_curator.stages.interleaved.io.readers.parquet import InterleavedParquetReaderStage
from nemo_curator.stages.interleaved.io.writers.tabular import InterleavedParquetWriterStage

if TYPE_CHECKING:
    from nemo_curator.pipeline.workflow import WorkflowRunResult
    from nemo_curator.tasks import Task

GIT_EXECUTABLE = "/usr/bin/git"


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


def _write_manifest(args: argparse.Namespace, metadata: dict[str, Any], tasks: list[Task] | None = None) -> None:
    output_path = Path(args.manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {key: value for key, value in vars(args).items() if key not in {"func"}}
    payload = {
        "stage": args.command,
        "parameters": parameters,
        "metadata": metadata,
        "num_output_tasks": len(tasks or []),
        **_code_state(),
    }
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary_path, output_path)


def _ray_data_executor(args: argparse.Namespace) -> RayDataExecutor:
    return RayDataExecutor(ignore_head_node=args.ignore_head_node)


@contextlib.contextmanager
def _persistent_ray_cluster() -> Iterator[None]:
    """Start a persistent Ray cluster for the duration of a GPU dedup workflow.

    The exact/fuzzy workflows create a *detached* ID-generator actor and then
    call ``ray.shutdown()`` from the driver. A detached actor only survives that
    shutdown when Ray is a standalone cluster (a separate ``ray start`` process),
    not a driver-local instance spun up by a bare ``ray.init()``. Without this,
    the executor's later ``ray.init()`` reconnects to nothing and the setup fails
    with "Did not find a valid ID generator actor".

    ``RayClient``/``SlurmRayClient`` start such a cluster and export
    ``RAY_ADDRESS``, so every subsequent ``ray.init()`` (in the workflow and the
    executor) attaches to the same long-lived cluster. If ``RAY_ADDRESS`` is
    already set (e.g. an externally managed cluster) the client attaches to it
    and leaves it running.
    """
    from nemo_curator.core.client import RayClient, SlurmRayClient

    client_kwargs: dict[str, Any] = {"include_dashboard": False}
    ray_temp_dir = os.environ.get("RAY_TMPDIR")
    if ray_temp_dir:
        client_kwargs["ray_temp_dir"] = ray_temp_dir

    client_cls = SlurmRayClient if os.environ.get("SLURM_JOB_ID") else RayClient
    client = client_cls(**client_kwargs)
    client.start()
    try:
        yield
    finally:
        client.stop()


def _partitioning_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if args.files_per_partition is not None:
        return {"files_per_partition": args.files_per_partition, "blocksize": None}
    return {"files_per_partition": None, "blocksize": args.input_blocksize}


def _dataset_files(path: str, extensions: list[str] | None = None) -> list[str]:
    """List dataset files under ``path`` while ignoring sidecar/metadata files.

    The Nemotron Parse output directory keeps the interleaved data as flat
    ``<hash>.parquet`` files alongside operational sidecars such as
    ``_perf_stats_<jobid>.parquet`` and ``_manifest_remaining.jsonl``. Those
    sidecars share the ``.parquet`` extension but do not carry the interleaved
    schema, so we drop any basename starting with ``_`` or ``.`` (the same
    convention PyArrow datasets, Spark, and Dask use) before handing the list to
    the partitioning stages.
    """
    from nemo_curator.utils.file_utils import get_all_file_paths_under

    files = get_all_file_paths_under(
        path,
        recurse_subdirectories=True,
        keep_extensions=extensions or [".parquet"],
    )
    return [file for file in files if not os.path.basename(file).startswith(("_", "."))]


def run_sha_inventory(args: argparse.Namespace) -> None:
    pipeline = Pipeline(
        name="pdf_sha256_inventory",
        stages=[
            FilePartitioningStage(
                file_paths=_dataset_files(args.input_path),
                file_extensions=[".parquet"],
                **_partitioning_kwargs(args),
            ),
            PdfSha256InventoryStage(
                output_path=args.output_path,
                pdf_root=args.pdf_root,
                sample_id_field=args.sample_id_field,
                pdf_name_field=args.pdf_name_field,
                validate_sample_pdf_mapping=not args.allow_sample_pdf_mismatch,
            ),
        ],
    )
    started = time.time()
    tasks = pipeline.run(executor=_ray_data_executor(args))
    metadata = {
        "runtime_seconds": time.time() - started,
        "num_samples": sum(task._metadata.get("num_samples", 0) for task in tasks or []),
        "num_hash_errors": sum(task._metadata.get("num_hash_errors", 0) for task in tasks or []),
        "num_resumed_tasks": sum(bool(task._metadata.get("resumed")) for task in tasks or []),
    }
    _write_manifest(args, metadata, tasks)


def run_sample_id_removal(args: argparse.Namespace) -> None:
    pipeline = Pipeline(
        name="interleaved_sample_id_removal",
        stages=[
            FilePartitioningStage(
                file_paths=_dataset_files(args.input_path),
                file_extensions=[".parquet"],
                **_partitioning_kwargs(args),
            ),
            InterleavedParquetReaderStage(read_kwargs={}),
            InterleavedSampleIdRemovalStage(
                ids_to_remove_path=args.ids_to_remove_path,
                sample_id_field=args.sample_id_field,
                duplicate_id_field=args.duplicate_id_field,
            ),
            InterleavedParquetWriterStage(
                path=args.output_path,
                mode="ignore",
                materialize_on_write=False,
            ),
        ],
    )
    started = time.time()
    tasks = pipeline.run(executor=_ray_data_executor(args))
    metadata = {
        "runtime_seconds": time.time() - started,
        "num_samples_in": sum(task._metadata.get("num_samples_in", 0) for task in tasks or []),
        "num_samples_removed": sum(task._metadata.get("num_samples_removed", 0) for task in tasks or []),
        "num_samples_out": sum(task._metadata.get("num_samples_out", 0) for task in tasks or []),
        "num_rows_in": sum(task._metadata.get("num_rows_in", 0) for task in tasks or []),
        "num_rows_out": sum(task._metadata.get("num_rows_out", 0) for task in tasks or []),
    }
    _write_manifest(args, metadata, tasks)


def run_exact_identification(args: argparse.Namespace) -> None:
    from nemo_curator.stages.interleaved.deduplication.exact_workflow import (
        InterleavedExactDeduplicationWorkflow,
    )

    workflow = InterleavedExactDeduplicationWorkflow(
        input_path=_dataset_files(args.input_path),
        output_path=args.output_path,
        input_blocksize=args.input_blocksize,
        identification_batchsize=args.identification_batchsize,
        text_mode="text_rows",
        text_separator=args.text_separator,
        total_nparts=args.total_nparts,
        rmm_pool_size=args.rmm_pool_size,
        spill_memory_limit=args.spill_memory_limit,
    )
    with _persistent_ray_cluster():
        result = workflow.run(executor=RayActorPoolExecutor(ignore_head_node=args.ignore_head_node))
    _write_manifest(args, result.metadata, result.pipeline_tasks.get("identification"))


def run_fuzzy_identification(args: argparse.Namespace) -> None:
    from nemo_curator.stages.interleaved.deduplication.fuzzy_workflow import (
        InterleavedTextFuzzyDeduplicationWorkflow,
    )

    workflow = InterleavedTextFuzzyDeduplicationWorkflow(
        input_path=_dataset_files(args.input_path),
        cache_path=args.cache_path,
        output_path=args.output_path,
        interleaved_text_mode="text_rows",
        input_blocksize=args.input_blocksize,
        input_files_per_partition=args.files_per_partition,
        text_separator=args.text_separator,
        perform_removal=False,
        seed=args.seed,
        char_ngrams=args.char_ngrams,
        num_bands=args.num_bands,
        minhashes_per_band=args.minhashes_per_band,
        bands_per_iteration=args.bands_per_iteration,
        lsh_num_output_partitions=args.lsh_num_output_partitions,
        lsh_rmm_pool_size=args.rmm_pool_size,
        lsh_spill_memory_limit=args.spill_memory_limit,
    )
    with _persistent_ray_cluster():
        result = workflow.run(executor=RayActorPoolExecutor(ignore_head_node=args.ignore_head_node))
    output_tasks = result.pipeline_tasks.get("connected_components") or result.pipeline_tasks.get("lsh")
    minhash_tasks = result.pipeline_tasks.get("minhash") or []
    metadata = {
        **result.metadata,
        "num_input_rows": sum(
            task._metadata.get("interleaved_minhash_metrics", {}).get("num_input_rows", 0) for task in minhash_tasks
        ),
        "num_input_samples": sum(task._metadata.get("num_documents", 0) for task in minhash_tasks),
        "num_hashable_samples": sum(task._metadata.get("num_hashable_documents", 0) for task in minhash_tasks),
    }
    _write_manifest(args, metadata, output_tasks)


def run_generated_id_removal(args: argparse.Namespace) -> None:
    workflow = InterleavedDuplicatesRemovalWorkflow(
        input_path=_dataset_files(args.input_path),
        ids_to_remove_path=args.ids_to_remove_path,
        output_path=args.output_path,
        input_files_per_partition=args.files_per_partition,
        input_blocksize=args.input_blocksize,
        input_file_extensions=[".parquet"],
        id_generator_path=args.id_generator_path,
        drop_id_field=True,
        output_mode="ignore",
        materialize_on_write=False,
    )
    # The removal workflow loads the ID generator into a detached actor, so it needs the
    # same persistent Ray cluster as the identification workflows (see _persistent_ray_cluster).
    with _persistent_ray_cluster():
        result: WorkflowRunResult = workflow.run(executor=_ray_data_executor(args))
    tasks = result.pipeline_tasks.get("removal")
    metadata = {
        **result.metadata,
        "num_samples_in": sum(task._metadata.get("num_samples_in", 0) for task in tasks or []),
        "num_samples_removed": sum(task._metadata.get("num_samples_removed", 0) for task in tasks or []),
        "num_samples_out": sum(task._metadata.get("num_samples_out", 0) for task in tasks or []),
        "num_rows_in": sum(task._metadata.get("num_rows_in", 0) for task in tasks or []),
        "num_rows_out": sum(task._metadata.get("num_rows_out", 0) for task in tasks or []),
    }
    _write_manifest(args, metadata, tasks)


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--input-blocksize", default="1GiB")
    parser.add_argument("--files-per-partition", type=int)
    parser.add_argument("--ignore-head-node", action="store_true")


def _add_gpu_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rmm-pool-size", default="auto")
    parser.add_argument("--spill-memory-limit", default="auto")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sha = subparsers.add_parser("sha-inventory")
    _add_execution_args(sha)
    sha.add_argument("--pdf-root", required=True)
    sha.add_argument("--output-path", required=True)
    sha.add_argument("--sample-id-field", default="sample_id")
    sha.add_argument("--pdf-name-field", default="pdf_name")
    sha.add_argument("--allow-sample-pdf-mismatch", action="store_true")
    sha.set_defaults(func=run_sha_inventory)

    sample_remove = subparsers.add_parser("sample-id-remove")
    _add_execution_args(sample_remove)
    sample_remove.add_argument("--ids-to-remove-path", required=True)
    sample_remove.add_argument("--output-path", required=True)
    sample_remove.add_argument("--sample-id-field", default="sample_id")
    sample_remove.add_argument("--duplicate-id-field", default="sample_id")
    sample_remove.set_defaults(func=run_sample_id_removal)

    exact = subparsers.add_parser("exact-identify")
    _add_execution_args(exact)
    _add_gpu_args(exact)
    exact.add_argument("--output-path", required=True)
    exact.add_argument("--identification-batchsize", type=int, default=1)
    exact.add_argument("--total-nparts", type=int)
    exact.add_argument("--text-separator", default="\n\n")
    exact.set_defaults(func=run_exact_identification)

    fuzzy = subparsers.add_parser("fuzzy-identify")
    _add_execution_args(fuzzy)
    _add_gpu_args(fuzzy)
    fuzzy.add_argument("--cache-path", required=True)
    fuzzy.add_argument("--output-path", required=True)
    fuzzy.add_argument("--seed", type=int, default=42)
    fuzzy.add_argument("--char-ngrams", type=int, default=24)
    fuzzy.add_argument("--num-bands", type=int, default=20)
    fuzzy.add_argument("--minhashes-per-band", type=int, default=13)
    fuzzy.add_argument("--bands-per-iteration", type=int, default=5)
    fuzzy.add_argument("--lsh-num-output-partitions", type=int)
    fuzzy.add_argument("--text-separator", default="\n\n")
    fuzzy.set_defaults(func=run_fuzzy_identification)

    generated_remove = subparsers.add_parser("generated-id-remove")
    _add_execution_args(generated_remove)
    generated_remove.add_argument("--ids-to-remove-path", required=True)
    generated_remove.add_argument("--id-generator-path", required=True)
    generated_remove.add_argument("--output-path", required=True)
    generated_remove.set_defaults(func=run_generated_id_removal)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
