#!/usr/bin/env python
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

r"""Run MinerU-HTML over the LaTeXML Parquet pool to get Markdown.

Adapts ``tutorials/text/mineru-html-extraction/run_pipeline.py`` to this dataset.
Three differences from that tutorial, each forced by something real:

**A tier filter, before the extractor.**  MinerU's cost is LLM inference, and
~29% of the pool is Tier C or rejected -- documents we would discard anyway.
``ParquetReader`` projects *columns* but applies no row predicate, so pointing it
at the pool would pay full inference on documents destined for the bin.  The
filter is a stage rather than a layout change because splitting part files by
tier would give files of ~6 documents, far too small to be worth the inodes.

**``--html-field html``.**  Our column is ``html``; the tutorial defaults to
``content`` because it was written for Common Crawl.

**``tier`` and ``arxiv_id`` are read explicitly.**  The tutorial reads only
``[html_field, url_field]``.  Anything not read is not written, so without this
the output would be Markdown with no way to attribute it to a paper -- and the
filter would have no column to filter on.

The input is LaTeXML output, not a scraped web page, so there is no navigation,
advertisement or cookie banner for the model to strip.  Its useful work here is
the HTML-to-Markdown serialization, and in particular recovering each formula
from the ``alttext`` LaTeX that ar5iv preserves beside the MathML.
"""

from __future__ import annotations

import argparse
import time

from loguru import logger

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.text.html_extraction import DEFAULT_MODEL, MinerUHtmlExtractor
from nemo_curator.stages.text.io.reader.parquet import ParquetReader
from nemo_curator.stages.text.io.writer.parquet import ParquetWriter
from nemo_curator.tasks import DocumentBatch

#: Tiers worth spending inference on.  A and B are the arXiv-strict "usable" set;
#: C is recoverable-in-principle but carries known residual-LaTeX artifacts.  All
#: three are extracted -- the judgement of whether C is good enough belongs
#: downstream, where the Markdown can actually be inspected, and re-running
#: inference later to add it back would cost more than including it now.  What is
#: never worth inference is ``rejected``: those rows carry null or empty ``html``.
DEFAULT_TIERS = ("A", "B", "C")


class HeadStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Truncate every reader partition to ``n`` rows, for a smoke run."""

    def __init__(self, n: int):
        self.n = n
        self.name = "head"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=batch.to_pandas().head(self.n),
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


class TierFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Drop documents whose quality tier is not wanted, before inference.

    Placed ahead of the extractor deliberately: filtering afterwards would give
    the same output for strictly more GPU time.
    """

    def __init__(self, tiers: tuple[str, ...] = DEFAULT_TIERS, tier_field: str = "tier"):
        self.tiers = set(tiers)
        self.tier_field = tier_field
        self.name = "tier_filter"

    def process(self, batch: DocumentBatch) -> DocumentBatch:
        frame = batch.to_pandas()
        kept = frame[frame[self.tier_field].isin(self.tiers)]
        return DocumentBatch(
            dataset_name=batch.dataset_name,
            data=kept,
            _metadata=batch._metadata,
            _stage_perf=batch._stage_perf,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Pool html/ directory, or a view's part files")
    ap.add_argument("--output", required=True)
    ap.add_argument("--server-url", required=True, help="OpenAI-compatible vLLM endpoint")
    ap.add_argument("--tiers", default=",".join(DEFAULT_TIERS), help="Comma-separated tiers to keep; empty keeps all")
    ap.add_argument("--html-field", default="html")
    ap.add_argument("--url-field", default="url")
    ap.add_argument("--served-model-name", default="mineru")
    ap.add_argument("--server-concurrency", type=int, default=48)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--output-format", default="mm_md", choices=["mm_md", "md", "json", "txt", "none"])
    ap.add_argument("--fallback", default="trafilatura", choices=["trafilatura", "bypass", "empty"])
    ap.add_argument("--blocksize", default="256MB")
    # The pool is 407 MB across 415 part files, so the default blocksize would
    # put the whole dataset in two partitions and two thirds of the cluster's
    # cores would idle.  Curator never splits a file across workers, so the
    # partitioning granularity is a file count, not a byte count.
    ap.add_argument("--files-per-partition", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Keep at most this many rows per reader partition")
    ap.add_argument("--simplify-workers", type=int, default=None)
    ap.add_argument("--inference-workers", type=int, default=None)
    ap.add_argument("--extract-workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tiers = tuple(t for t in args.tiers.split(",") if t)

    reader = ParquetReader(
        file_paths=args.input,
        blocksize=args.blocksize if args.files_per_partition is None else None,
        files_per_partition=args.files_per_partition,
        # tier drives the filter; arxiv_id and shard keep the output joinable
        # back to the pool.  Columns not listed here never reach the writer.
        fields=[args.html_field, args.url_field, "tier", "arxiv_id", "shard"],
        # Arrow-backed large string columns overflow 32-bit offsets when a
        # partition is pickled between stages; object dtype has no such limit.
        read_kwargs={"dtype_backend": "numpy_nullable"},
    )

    pipeline = Pipeline(name="arxiv_latexml_mineru", description="LaTeXML HTML -> Markdown via MinerU-HTML")
    pipeline.add_stage(reader)
    if args.limit:
        pipeline.add_stage(HeadStage(args.limit))
    if tiers:
        pipeline.add_stage(TierFilter(tiers))
    pipeline.add_stage(
        MinerUHtmlExtractor(
            base_url=args.server_url,
            served_model_name=args.served_model_name,
            server_concurrency=args.server_concurrency,
            html_field=args.html_field,
            url_field=args.url_field,
            model_identifier=args.model,
            max_model_len=args.max_model_len,
            output_format=args.output_format,
            fallback=args.fallback,
            simplify_workers=args.simplify_workers,
            inference_workers=args.inference_workers,
            extract_workers=args.extract_workers,
        )
    )
    pipeline.add_stage(ParquetWriter(path=args.output, mode="overwrite" if args.overwrite else "ignore"))

    logger.info(pipeline.describe())
    started = time.perf_counter()
    pipeline.run()
    logger.info(f"done in {time.perf_counter() - started:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
