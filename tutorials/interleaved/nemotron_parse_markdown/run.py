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

r"""PDFs to interleaved markdown, in two phases.

Needs a running Ray cluster: the executor schedules every stage as Ray actors,
and the parse phase needs a GPU.

**Phase 1 -- parse.**  :class:`NemotronParsePDFReader` renders each page, runs
Nemotron-Parse on it, and decodes the output into one row per element: the
text, the layout class the model gave it, its page and its bounding box.  That
is the *Nemotron-Parse element format*, and it is the artifact worth keeping,
because reproducing it costs a GPU-hour per few hundred documents.

**Phase 2 -- post-process.**  :class:`NemotronParseMarkdownPostprocessor` turns
those elements into a document: paragraphs rejoined across columns and pages,
each figure next to its caption, running heads and bibliographies dropped, text
written as markdown, pictures still in place in the reading order.  CPU-only,
minutes rather than hours, and the half whose rules you actually want to tune.

Which is why ``--phase`` exists.  Run ``parse`` once and keep its output; then
run ``postprocess`` over that output as often as you like, changing the rules
each time, without touching a GPU.  ``all`` chains both in one pipeline and
never writes the intermediate, which is what you want for a corpus you have
already settled the rules for.

Example -- both phases, three PDFs, straight to markdown::

    python run.py --phase all --pdf-dir /data/pdfs --manifest manifest.jsonl \
        --output-dir ./out --max-pdfs 3

Example -- parse once, keeping the elements::

    python run.py --phase parse --pdf-dir /data/pdfs --manifest manifest.jsonl \
        --output-dir /data/elements

Example -- then iterate on the rules, no GPU::

    python run.py --phase postprocess --input-dir /data/elements \
        --output-dir /data/markdown-v2 --no-skip-toc-bib --min-caption-chars 60

Example -- a later Nemotron-Parse release with the same output contract.  An
unregistered version needs its weights named, because guessing a HuggingFace id
is worse than asking for one; a registered one names its own::

    python run.py --phase parse --pdf-dir /data/pdfs --manifest manifest.jsonl \
        --output-dir /data/elements \
        --parse-version v2.0 --model-path nvidia/NVIDIA-Nemotron-Parse-v2.0

Example -- what the rules threw away, for a viewer rather than for training::

    python run.py --phase postprocess --input-dir /data/elements \
        --output-dir /data/audit --emit-dropped

Creating a manifest for a directory of PDFs::

    for f in /data/pdfs/*.pdf; do
        echo "{\"file_name\": \"$(basename "$f")\"}" >> manifest.jsonl
    done
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetReader, InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse import (
    Config,
    NemotronParseMarkdownPostprocessor,
    NemotronParsePDFReader,
)

if TYPE_CHECKING:
    from nemo_curator.tasks import Task

PARSE = "parse"
POSTPROCESS = "postprocess"
ALL = "all"

#: Phase 2 reads a whole document at a time -- paragraph reconstitution reaches
#: across pages -- so a document must not be split between two batches.  The
#: parse phase emits a document's rows together and the Parquet writer keeps a
#: task's rows in one file, so one file per task preserves it.
FILES_PER_PARTITION = 1


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Assemble the phases ``--phase`` asked for."""
    pipeline = Pipeline(
        name=f"nemotron_parse_markdown_{args.phase}",
        description="PDF -> Nemotron-Parse elements -> interleaved markdown",
    )

    # 1. Phase 1: PDFs in, one row per element the model found.
    if args.phase in (PARSE, ALL):
        pipeline.add_stage(
            NemotronParsePDFReader(
                manifest_path=args.manifest,
                zip_base_dir=args.zip_base_dir,
                pdf_dir=args.pdf_dir,
                jsonl_base_dir=args.jsonl_base_dir,
                model_path=args.model_path,
                parse_version=args.parse_version,
                backend=args.backend,
                pdfs_per_task=args.pdfs_per_task,
                max_pdfs=args.max_pdfs,
                dpi=args.dpi,
                max_pages=args.max_pages,
                inference_batch_size=args.inference_batch_size,
                max_num_seqs=args.max_num_seqs,
                text_in_pic=args.text_in_pic,
                enforce_eager=args.enforce_eager,
                min_crop_px=args.min_crop_size,
                dataset_name=args.dataset_name,
            )
        )
    else:
        # ...or read a previous parse phase's output back off disk.
        pipeline.add_stage(
            InterleavedParquetReader(
                file_paths=args.input_dir,
                files_per_partition=FILES_PER_PARTITION,
            )
        )

    # 2. Phase 2: elements in, a document out.
    if args.phase in (POSTPROCESS, ALL):
        pipeline.add_stage(
            NemotronParseMarkdownPostprocessor(
                config=build_config(args),
                emit_dropped=args.emit_dropped,
                fuse=args.fuse,
            )
        )

    # 3. Whatever the last phase produced.
    pipeline.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            mode=args.write_mode,
            materialize_on_write=False,
        )
    )
    return pipeline


def build_config(args: argparse.Namespace) -> Config:
    """The post-processing rules, from the flags that switch them off."""
    return Config(
        min_caption_chars=args.min_caption_chars,
        min_caption_words=args.min_caption_words,
        non_bib_toc_words=args.non_bib_toc_words,
        strip_markdown=args.strip_markdown,
        assign_floats=not args.no_assign_floats,
        reconstitute_paragraphs=not args.no_reconstitute_paragraphs,
        skip_toc_bib=not args.no_skip_toc_bib,
        drop_page_furniture=not args.keep_page_furniture,
        check_repeated_words=not args.no_check_repeated_words,
        require_tabular_in_tables=not args.no_require_tabular,
        keep_images=not args.text_only,
        drop_classes=frozenset(args.drop_class),
    )


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def _git(*command: str) -> str | None:
    """Ask git something about this checkout, or ``None`` if it cannot say.

    ``None`` is displayed as *not recorded*, which is true and useful; a
    plausible-looking wrong commit or a repo URL that is not the one the code
    came from is neither.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", *command],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_atlas_manifest(args: argparse.Namespace) -> None:
    """Record what this run produced, in the directory it produced it in.

    Written by the run that made the data rather than from memory afterwards,
    which is the only version of this that is worth anything.  ``command`` is
    ``sys.argv`` verbatim: a command shortened for readability is a command
    nobody can re-run.
    """
    if not args.atlas_id:
        return

    step, description, granularity = {
        PARSE: (
            "parse",
            "One row per element Nemotron-Parse found in a PDF: text, layout class, page, bbox.",
            "one row per page element",
        ),
        POSTPROCESS: (
            "postprocess",
            "The same elements assembled into a document and written as markdown.",
            "one row per markdown block or image, in reading order",
        ),
        ALL: (
            "parse-and-postprocess",
            "PDFs parsed and assembled into interleaved markdown in one pass.",
            "one row per markdown block or image, in reading order",
        ),
    }[args.phase]

    commit = _git("rev-parse", "HEAD")
    if commit and _git("status", "--porcelain"):
        # The tree has uncommitted changes, so this commit does not describe the
        # code that ran. Saying so is the whole point: a sha that resolves to
        # something other than what produced the data is worse than no sha,
        # because it looks checkable. `-dirty` matches how the snapshot
        # directories in this project already mark the same situation.
        commit = f"{commit}-dirty"
        logger.warning(f"working tree is dirty; recording commit as {commit}")

    manifest: dict[str, Any] = {
        "id": args.atlas_id,
        "title": args.atlas_title or args.atlas_id,
        "description": description,
        "format": "parquet",
        "granularity": granularity,
        "parents": [{"id": parent} for parent in args.atlas_parent],
        "edge": {
            "name": step,
            "from": args.atlas_parent[0] if args.atlas_parent else None,
            "to": args.atlas_id,
            "repo": _git("remote", "get-url", "origin"),
            "commit": commit,
            "command": args.atlas_command or shlex.join([sys.executable, *sys.argv]),
        },
    }

    path = os.path.join(args.output_dir, ".atlas.json")
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    logger.info(f"Wrote {path}")


def _write_perf_summary(results: list[Task], output_dir: str, wall_time: float) -> None:
    """Per-stage timings, so a slow phase can be told from a slow stage.

    Written as JSONL, deliberately.  ``--phase parse`` writes its elements into
    this same directory and ``--phase postprocess`` reads that directory back
    as one Parquet dataset; a stats file in Parquet would be picked up as data
    and fail the read on a schema it never shared.
    """
    valid = [r for r in results if r is not None] if results else []
    if not valid:
        logger.warning("No results to write perf summary for")
        return
    if len(valid) < len(results):
        logger.warning(f"{len(results) - len(valid)} tasks returned None (failed)")

    rows = [
        {
            "task_id": task.task_id,
            "stage_name": perf.stage_name,
            "process_time_s": perf.process_time,
            "actor_idle_time_s": perf.actor_idle_time,
            "num_items_processed": perf.num_items_processed,
            **{f"custom_{k}": v for k, v in perf.custom_metrics.items()},
        }
        for task in valid
        for perf in task._stage_perf
    ]
    job_id = os.environ.get("SLURM_JOB_ID", f"local_{int(time.time())}")
    perf_path = os.path.join(output_dir, f"_perf_stats_{job_id}.jsonl")
    with open(perf_path, "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)

    by_stage: dict[str, list[float]] = {}
    items: dict[str, float] = {}
    for row in rows:
        by_stage.setdefault(row["stage_name"], []).append(row["process_time_s"])
        items[row["stage_name"]] = items.get(row["stage_name"], 0) + (row["num_items_processed"] or 0)

    logger.info(f"\n{'=' * 70}\n  PERFORMANCE SUMMARY  (wall_time={wall_time:.1f}s, tasks={len(valid)})\n{'=' * 70}")
    for stage_name, times in by_stage.items():
        ordered = sorted(times)
        p50 = ordered[len(ordered) // 2]
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
        logger.info(
            f"  {stage_name:44s}  avg={sum(times) / len(times):8.2f}s  p50={p50:8.2f}s  "
            f"p95={p95:8.2f}s  sum={sum(times):10.1f}s  items={items[stage_name]:.0f}"
        )
    logger.info(f"{'=' * 70}\n")


def main(args: argparse.Namespace) -> None:
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.manifest_only:
        # A Slurm array writes its data from many tasks and its provenance
        # once, afterwards, from a job that knows the whole run succeeded.
        # Starting Ray to write one JSON file would be silly.
        write_atlas_manifest(args)
        return

    if os.environ.get("SLURM_JOB_ID"):
        from nemo_curator.core.client import SlurmRayClient

        ray_client = SlurmRayClient()
    else:
        ray_client = RayClient()
    ray_client.start()

    try:
        pipeline = build_pipeline(args)
        logger.info(f"\n{pipeline.describe()}")

        executor = XennaExecutor(
            config={
                "execution_mode": args.execution_mode,
                "ignore_failures": True,
                "failures_return_nones": True,
                "reset_workers_on_failure": True,
            }
        )

        started = time.perf_counter()
        results = pipeline.run(executor=executor, checkpoint_path=args.checkpoint_dir)
        wall_time = time.perf_counter() - started

        valid = sum(1 for r in results if r is not None)
        logger.info(f"Pipeline finished in {wall_time:.1f}s, {valid} output tasks ({len(results) - valid} failed)")
        _write_perf_summary(results, args.output_dir, wall_time)
        write_atlas_manifest(args)
    finally:
        ray_client.stop()


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Each phase needs its own input, and only its own."""
    if args.manifest_only:
        if not args.atlas_id:
            parser.error("--manifest-only needs --atlas-id: there is nothing to record without one")
        return
    if args.phase in (PARSE, ALL):
        if not args.manifest:
            parser.error("--manifest is required for the parse phase")
        if not (args.pdf_dir or args.zip_base_dir or args.jsonl_base_dir):
            parser.error("the parse phase needs one of --pdf-dir, --zip-base-dir or --jsonl-base-dir")
    elif not args.input_dir:
        parser.error("--input-dir is required for the postprocess phase")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDFs to interleaved markdown via Nemotron-Parse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Which half to run
    parser.add_argument(
        "--phase",
        default=ALL,
        choices=[ALL, PARSE, POSTPROCESS],
        help="'parse' writes the element format and stops; 'postprocess' reads it back; 'all' chains both",
    )

    # I/O
    parser.add_argument("--manifest", help="JSONL manifest listing PDFs (parse phase)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pdf-dir", help="Directory of .pdf files")
    source.add_argument("--zip-base-dir", help="Root of a CC-MAIN zip archive hierarchy")
    source.add_argument("--jsonl-base-dir", help="Root of a JSONL-based PDF dataset")
    parser.add_argument("--input-dir", help="Parquet written by an earlier parse phase (postprocess phase)")
    parser.add_argument("--output-dir", required=True, help="Where to write parquet")
    parser.add_argument("--dataset-name", default="pdf_dataset", help="Name assigned to output tasks")
    parser.add_argument(
        "--write-mode",
        default="ignore",
        choices=["ignore", "overwrite", "append", "error"],
        help="What to do when the output directory already has data in it",
    )

    # Model
    parser.add_argument(
        "--model-path",
        default=None,
        help="HuggingFace ID or local path. Unset, the weights named by --parse-version are used",
    )
    parser.add_argument(
        "--parse-version",
        default=None,
        help="Which release's behaviour to assume, e.g. v1.2. Recognised from --model-path when omitted",
    )
    parser.add_argument("--backend", default="vllm", choices=["hf", "vllm"], help="Inference backend")
    parser.add_argument("--text-in-pic", action="store_true", help="Predict text inside pictures (v1.2+)")

    # Parsing
    parser.add_argument("--pdfs-per-task", type=int, default=10, help="PDFs per processing task")
    parser.add_argument("--max-pdfs", type=int, default=None, help="Limit total PDFs (for a smoke test)")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering resolution")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per PDF")
    parser.add_argument("--min-crop-size", type=int, default=10, help="Min pixel dimension for an image crop")
    parser.add_argument("--inference-batch-size", type=int, default=4, help="Pages per GPU pass (HF only)")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="Max concurrent sequences (vLLM only)")
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graph capture: no ~35min compile at startup, slightly less throughput",
    )

    # Post-processing: every rule can be switched off, which is what makes a
    # before/after comparison mean anything.
    parser.add_argument("--min-caption-chars", type=int, default=100, help="Shorter captions are layout noise")
    parser.add_argument("--min-caption-words", type=int, default=10, help="...and so are sparser ones")
    parser.add_argument(
        "--non-bib-toc-words", type=int, default=35, help="Prose this substantial ends a bibliography skip"
    )
    parser.add_argument("--drop-class", action="append", default=[], help="Drop this element class (repeatable)")
    parser.add_argument("--strip-markdown", action="store_true", help="Reduce markdown to plain text first")
    parser.add_argument("--no-assign-floats", action="store_true", help="Leave floats in the reading flow")
    parser.add_argument("--no-reconstitute-paragraphs", action="store_true", help="Do not rejoin broken sentences")
    parser.add_argument("--no-skip-toc-bib", action="store_true", help="Keep the contents and the bibliography")
    parser.add_argument("--keep-page-furniture", action="store_true", help="Keep running heads and page numbers")
    parser.add_argument("--no-check-repeated-words", action="store_true", help="Keep degenerate repeated blocks")
    parser.add_argument("--no-require-tabular", action="store_true", help="Keep Tables with no tabular environment")
    parser.add_argument("--text-only", action="store_true", help="Drop pictures instead of interleaving them")
    parser.add_argument("--emit-dropped", action="store_true", help="Keep condemned rows, marked with the reason")
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Run the six post-processing steps as one stage: faster, but the intermediate is not observable",
    )

    # Execution
    parser.add_argument(
        "--execution-mode", default="streaming", choices=["streaming", "batch"], help="XennaExecutor mode"
    )

    # Scale.  Curator splits the input across a Slurm array itself, by hashing
    # each source task -- set nothing here, just submit with --array and it
    # reads SLURM_ARRAY_TASK_ID/COUNT.  --checkpoint-dir is what makes a
    # requeued task skip the shards the last attempt finished.
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Resumability directory (local filesystem). Completed input shards are skipped on rerun",
    )

    # Provenance
    parser.add_argument("--atlas-id", help="Dataset id to record in .atlas.json, e.g. arxiv/pdf-markdown/2026-08-21")
    parser.add_argument("--atlas-title", help="Human name for the dataset")
    parser.add_argument("--atlas-parent", action="append", default=[], help="Parent dataset id (repeatable)")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write .atlas.json for --output-dir and exit, running no pipeline. For recording "
        "provenance after a Slurm array, once the whole run is known to have succeeded",
    )
    parser.add_argument(
        "--atlas-command",
        default=None,
        help="Record this as the producing command instead of this process's argv. For a Slurm "
        "array, the reproducible command is the sbatch that launched all of it, not one task's share",
    )

    parsed = parser.parse_args()
    _validate(parser, parsed)
    main(parsed)
