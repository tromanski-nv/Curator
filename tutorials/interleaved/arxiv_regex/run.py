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

r"""arXiv LaTeX -> interleaved Parquet pipeline.

Reads raw arXiv bulk source shards (``arXiv_src_YYMM_NNN.tar``), parses each
submission's LaTeX, and writes one row per document element -- body text,
figure, caption -- in reading order.

This script needs a running Ray cluster: ``RayClient().start()`` brings one up
locally, so it will not run on a login node or anywhere ``ulimit -u`` /
``ulimit -v`` are tight.  To exercise a single stage without Ray, construct the
stage directly and call ``stage.process(task)``.

Example -- one shard to Parquet::

    python run.py \\
        --input-dir /data/arxiv-src/ \\
        --output-dir /data/arxiv-interleaved/ \\
        --mode overwrite

Example -- smoke test on the first 200 papers of each shard::

    python run.py \\
        --input-dir /data/arxiv-src/ \\
        --output-dir /tmp/arxiv_smoke/ \\
        --max-papers-per-tar 200 \\
        --papers-per-task 50 \\
        --mode overwrite

Example -- text-only pass (no figure bytes, captions kept)::

    python run.py \\
        --input-dir /data/arxiv-src/ \\
        --output-dir /data/arxiv-text/ \\
        --text-only \\
        --min-text-chars 200 \\
        --mode overwrite

Example -- keep only figures a browser can render (drops ~96% of pre-2005
figures, which are PostScript)::

    python run.py \\
        --input-dir /data/arxiv-src/ \\
        --output-dir /data/arxiv-raster/ \\
        --image-content-types image/png image/jpeg \\
        --mode overwrite

Example -- read the shards straight out of arXiv's requester-pays bucket::

    python run.py \\
        --input-dir s3://arxiv/src/ \\
        --output-dir /data/arxiv-interleaved/ \\
        --storage-options-json '{"requester_pays": true}' \\
        --mode overwrite
"""

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from nemo_curator.backends.ray_data import RayDataExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.latex.arxiv.regex import ArxivLatexReader
from nemo_curator.stages.interleaved.stages import BaseInterleavedFilterStage
from nemo_curator.tasks.interleaved import InterleavedBatch


@dataclass
class ShortTextFilterStage(BaseInterleavedFilterStage):
    """Drop body-text rows shorter than ``min_chars`` characters.

    Captions are exempt: a one-line caption is still worth keeping next to its
    figure, and dropping it would break the text/figure/caption triple.
    """

    min_chars: int = 200
    name: str = "short_text_filter"

    def content_keep_mask(self, _task: InterleavedBatch, df: pd.DataFrame) -> pd.Series:
        is_body_text = df["modality"] == "text"
        if "element_class" in df.columns:
            is_body_text &= df["element_class"] == "text"
        too_short = is_body_text & (df["text_content"].fillna("").str.len() < self.min_chars)
        return ~too_short


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
    out_options = args.output_storage_options_json
    write_kwargs = {"storage_options": out_options} if out_options else {}

    pipe = Pipeline(
        name="arxiv_latex_pipeline",
        description="arXiv source tars -> parsed LaTeX -> interleaved rows -> Parquet",
    )

    # 1. Read: FilePartitioningStage -> ArxivTarPartitioningStage -> ArxivLatexReaderStage.
    pipe.add_stage(
        ArxivLatexReader(
            file_paths=args.input_dir,
            files_per_partition=args.files_per_partition,
            papers_per_task=args.papers_per_task,
            max_papers_per_tar=args.max_papers_per_tar,
            include_images=not args.text_only,
            emit_captions=not args.no_captions,
            drop_bibliography=not args.keep_bibliography,
            drop_appendix=args.drop_appendix,
            max_image_bytes=args.max_image_bytes,
            image_content_types=tuple(args.image_content_types) if args.image_content_types else None,
            max_batch_bytes=args.max_batch_bytes,
            read_kwargs=read_kwargs,
        )
    )

    # 2. Optional filter: drop stub body-text runs (section headings, stray
    #    fragments between two floats).  0 disables the stage entirely.
    if args.min_text_chars > 0:
        pipe.add_stage(ShortTextFilterStage(min_chars=args.min_text_chars))

    # 3. Write.  Figure bytes are already materialized by the reader, so
    #    materialize_on_write has nothing left to fetch.
    pipe.add_stage(
        InterleavedParquetWriterStage(
            path=args.output_dir,
            materialize_on_write=False,
            mode=args.mode,
            on_materialize_error=args.on_materialize_error,
            write_kwargs=write_kwargs,
        )
    )

    return pipe


def main(args: argparse.Namespace) -> None:
    # Ray is required: the pipeline executor schedules every stage as Ray tasks.
    ray_client = RayClient()
    ray_client.start()
    pipeline = build_pipeline(args)
    print(pipeline.describe())
    pipeline.run(executor=RayDataExecutor())
    ray_client.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arXiv LaTeX source -> interleaved Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory (or single path / glob) of arXiv_src_YYMM_NNN.tar shards. Local or s3://.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write Parquet files into")
    parser.add_argument(
        "--mode",
        type=str,
        default="ignore",
        choices=["ignore", "overwrite", "append", "error"],
        help="Output directory handling mode",
    )

    # Partitioning
    parser.add_argument(
        "--files-per-partition", type=int, default=1, help="arXiv source tars per file-partitioning task"
    )
    parser.add_argument(
        "--papers-per-task",
        type=int,
        default=100,
        dest="papers_per_task",
        help="Submissions per reader task. Lower it if worker memory is tight; figure bytes dominate.",
    )
    parser.add_argument(
        "--max-papers-per-tar",
        type=int,
        default=None,
        dest="max_papers_per_tar",
        help="Index at most this many submissions per shard (smoke tests). None = all ~2364.",
    )
    parser.add_argument(
        "--max-batch-bytes",
        type=int,
        default=None,
        dest="max_batch_bytes",
        help="Split reader output into batches of roughly this size (never splitting a paper)",
    )

    # Content selection
    parser.add_argument(
        "--text-only",
        action="store_true",
        dest="text_only",
        help="Skip figure bytes entirely (include_images=False). Much faster and far smaller output.",
    )
    parser.add_argument(
        "--no-captions",
        action="store_true",
        dest="no_captions",
        help="Do not emit figure captions as their own text rows",
    )
    parser.add_argument(
        "--keep-bibliography",
        action="store_true",
        dest="keep_bibliography",
        help="Keep the bibliography instead of truncating the body at it",
    )
    parser.add_argument(
        "--drop-appendix", action="store_true", dest="drop_appendix", help=r"Truncate the body at \appendix"
    )
    parser.add_argument(
        "--image-content-types",
        nargs="*",
        default=None,
        dest="image_content_types",
        help=(
            "Keep only figures whose sniffed MIME type is listed, "
            "e.g. 'image/png image/jpeg'. Default keeps every format, including PostScript."
        ),
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=None,
        dest="max_image_bytes",
        help="Skip figures larger than this many bytes",
    )

    # Filtering
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=0,
        dest="min_text_chars",
        help="Drop body-text rows shorter than this many characters (0 = disable the filter stage)",
    )

    # Error handling
    parser.add_argument(
        "--on-materialize-error",
        type=str,
        default="drop_row",
        choices=["error", "warn", "drop_row", "drop_sample"],
        dest="on_materialize_error",
        help="What the writer does with rows carrying a materialize_error (set by on_missing_graphics='annotate')",
    )

    # Storage
    parser.add_argument(
        "--storage-options-json",
        type=_parse_json,
        default=None,
        help='JSON-encoded fsspec storage options for the *input* shards, e.g. \'{"requester_pays": true}\'',
    )
    parser.add_argument(
        "--output-storage-options-json",
        type=_parse_json,
        default=None,
        help="JSON-encoded fsspec storage options for the *output* directory (only needed for remote outputs)",
    )

    main(parser.parse_args())
