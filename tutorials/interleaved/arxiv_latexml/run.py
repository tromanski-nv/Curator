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

r"""Convert arXiv LaTeX source to HTML5 + presentation MathML with LaTeXML.

Reads ``arXiv_src_YYMM_NNN.tar`` shards from a local directory or any fsspec
URL, converts each submission with ``latexmlc``, and writes one Parquet row per
submission.

Requires the ``latexmlc`` binary on PATH -- see README.md.  Smoke test against
a handful of papers first::

    python run.py --input /data/arxiv-src --output /out/html \
        --limit 1 --max-papers-per-tar 8

Whole corpus, reading arXiv's requester-pays bucket directly::

    python run.py --input s3://arxiv/src --output /out/html \
        --storage-options-json '{"requester_pays": true}' \
        --snapshot snapshot-2026-07-27 --converter-id "$(cut -c1-12 image.sha256)"

Resume a run that stopped part way -- point it at its own output; submissions
already present are skipped, so this is also how a corpus run inherits a
sample's work rather than redoing it::

    python run.py --input /data/arxiv-src --output /out/html \
        --resume-from /out/html --mode append
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.latex.arxiv.latexml.convert import AR5IV_CONFIG, LatexmlConfig
from nemo_curator.stages.interleaved.latex.arxiv.latexml.stage import (
    LATEXMLC_INSTALL_HINT,
    ArxivLatexmlReader,
)
from nemo_curator.stages.text.io.writer import ParquetWriter


def _parse_json(raw: str) -> dict:
    """argparse type= callback: parse a JSON string into a dict."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        msg = f"must be valid JSON: {e}"
        raise argparse.ArgumentTypeError(msg) from e


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    # Input and output storage options are kept separate: the arXiv bucket is
    # requester-pays, and forwarding that option to a local write would fail.
    read_kwargs = {"storage_options": args.storage_options_json} if args.storage_options_json else {}
    write_kwargs = {"storage_options": args.output_storage_options_json} if args.output_storage_options_json else {}

    config = LatexmlConfig(
        executable=args.latexmlc,
        latexml_timeout_s=args.timeout,
    )

    pipe = Pipeline(
        name="arxiv_latexml_pipeline",
        description="arXiv source tars -> LaTeXML -> HTML5 + presentation MathML -> Parquet",
    )
    pipe.add_stage(
        ArxivLatexmlReader(
            file_paths=args.input,
            files_per_partition=args.files_per_partition,
            limit=args.limit,
            papers_per_task=args.papers_per_task,
            max_papers_per_tar=args.max_papers_per_tar,
            config=config,
            snapshot=args.snapshot,
            converter_id=args.converter_id,
            resume_from=args.resume_from,
            asset_dir=args.asset_dir,
            scratch_dir=args.scratch_dir,
            read_kwargs=read_kwargs,
        )
    )
    pipe.add_stage(
        ParquetWriter(
            path=args.output,
            mode=args.mode,
            write_kwargs=write_kwargs,
        )
    )
    return pipe


def main(args: argparse.Namespace) -> None:
    # Checked here as well as in the stage, so a typo in --latexmlc costs a
    # message rather than a scheduled pipeline that fails on every worker.
    if shutil.which(args.latexmlc) is None:
        print(LATEXMLC_INSTALL_HINT, file=sys.stderr)
        raise SystemExit(1)

    # Ray is required: the pipeline executor schedules every stage as Ray tasks.
    ray_client = RayClient()
    ray_client.start()
    pipeline = build_pipeline(args)
    print(pipeline.describe())
    pipeline.run(executor=RayDataExecutor())
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument(
        "--input",
        required=True,
        help="Directory or fsspec URL holding arXiv_src_*.tar shards (local path or s3://...)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to write Parquet into (local path, or a remote URL with --output-storage-options-json)",
    )
    parser.add_argument(
        "--mode",
        default="error",
        choices=["ignore", "overwrite", "append", "error"],
        help="What to do when --output already exists. Use 'append' when resuming (default: error)",
    )

    # Work sizing
    parser.add_argument(
        "--files-per-partition",
        type=int,
        default=1,
        help="Shards per partition. Shards are ~500 MB, so one per task is usually right (default: 1)",
    )
    parser.add_argument(
        "--papers-per-task",
        type=int,
        default=100,
        help="Submissions converted per task. Bounds peak memory and retry granularity (default: 100)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many shards (smoke tests)",
    )
    parser.add_argument(
        "--max-papers-per-tar",
        type=int,
        default=None,
        help="Index at most this many submissions per shard (smoke tests)",
    )

    # Converter
    parser.add_argument(
        "--latexmlc",
        default=AR5IV_CONFIG.executable,
        help="Path to the latexmlc binary (default: found on PATH)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=AR5IV_CONFIG.latexml_timeout_s,
        help=(
            "Per-document LaTeXML timeout in seconds. Measured: completion rate at 600s and at "
            "2700s is identical, so the extra 2100s is only paid by documents that fail anyway "
            f"(default: {AR5IV_CONFIG.latexml_timeout_s})"
        ),
    )

    # Provenance
    parser.add_argument(
        "--snapshot",
        default="",
        help="Source snapshot name, recorded on every row so a pool spanning snapshots stays separable",
    )
    parser.add_argument(
        "--converter-id",
        default="",
        help="Converter fingerprint (e.g. first 12 chars of the container image sha256), recorded per row",
    )

    # Resume and side output
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Directory of Parquet from an earlier run; submissions already present there are skipped",
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Write rasterized figures here, one tar per shard. Omitted by default",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Where each worker unpacks submissions. Defaults to the system temp directory",
    )

    # Storage
    parser.add_argument(
        "--storage-options-json",
        type=_parse_json,
        default=None,
        help='JSON fsspec storage options for the *input* shards, e.g. \'{"requester_pays": true}\'',
    )
    parser.add_argument(
        "--output-storage-options-json",
        type=_parse_json,
        default=None,
        help="JSON fsspec storage options for the *output* directory (only needed for remote outputs)",
    )

    main(parser.parse_args())
