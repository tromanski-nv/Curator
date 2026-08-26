# arXiv LaTeX → HTML with LaTeXML

Converts arXiv bulk source tarballs to HTML5 with presentation MathML by running
`latexmlc` over each submission, and writes one Parquet row per submission.

```
arXiv_src_*.tar
      │
      ▼
FilePartitioningStage            group shard paths into tasks
      │
      ▼
LatexmlTarPartitioningStage      read member headers only; emit byte ranges
      │  {"tar": ..., "member": ..., "offset": ..., "size": ...}
      ▼
LatexmlConvertStage              seek → gunzip → latexmlc → grade → scan
      │  DocumentBatch, one row per submission
      ▼
ParquetWriter
```

`ArxivLatexmlReader` is the `CompositeStage` wiring the first three together, so
you add one stage rather than three.

## Requirements

`latexmlc` must be on `PATH`. It is a Perl binary, not a Python package, so it
cannot be installed with pip. The public ar5iv image bundles LaTeXML with the
ar5iv bindings this tutorial expects:

```bash
docker pull latexml/ar5ivist:2512.17     # amd64 only; no aarch64 build exists
```

The bindings matter. `bindings/` carries LaTeXML implementations of packages
LaTeXML does not know natively, and `supported_originals/` carries real journal
classes (`revtex.cls`, `iopart.cls`, `elsart.cls`, `mn.sty`, …) that arXiv
submissions ship against. Without them, papers accumulate undefined-macro errors
and trip the 100-error fatal — `astro-ph/0301003` measured 16 errors without
them and 0 with.

Run the pipeline inside that image, or pass `--latexmlc /path/to/latexmlc` if
you have your own build.

## Running it

Smoke test first — one shard, eight papers:

```bash
python run.py --input /data/arxiv-src --output /out/html \
    --limit 1 --max-papers-per-tar 8
```

Whole corpus, reading arXiv's requester-pays bucket directly:

```bash
python run.py \
    --input s3://arxiv/src \
    --output /out/html \
    --storage-options-json '{"requester_pays": true}' \
    --snapshot snapshot-2026-07-27 \
    --converter-id "$(cut -c1-12 image.sha256)"
```

`--snapshot` and `--converter-id` are recorded on every row. They cost nothing
and are what keeps a dataset separable later: rows converted from different
source snapshots, or by different converter builds, stay distinguishable by
predicate instead of by remembering which directory came from which run.

## Resuming

Point a run at its own output. Submissions whose `source_sha256` is already
present are skipped:

```bash
python run.py --input /data/arxiv-src --output /out/html \
    --resume-from /out/html --mode append
```

Resume is derived from the written Parquet rather than from a ledger, because a
ledger can disagree with reality after a crash. The same mechanism lets a corpus
run inherit a sample run's work instead of redoing it — point `--resume-from` at
the sample's output.

## Output

One row per submission, including submissions that produced no HTML. A PDF-only
submission and a tarball with no root `.tex` both produce a row, so the
denominator stays "of all submissions" rather than silently becoming "of
submissions that converted".

| column | notes |
|---|---|
| `arxiv_id`, `url` | `url` survives a reader that projects to `[html, url]` |
| `html` | `NULL` when nothing converted |
| `status`, `tier` | see `latexml/model.py`; `tier` is `A`/`B`/`C`/`rejected` |
| `kind` | `tar`, `single_file`, `pdf_only`, `empty` |
| `n_warning`, `n_error`, `n_fatal` | LaTeXML diagnostic counts |
| `n_math`, `n_alttext`, `n_img`, `n_section` | structural counts of the HTML |
| `n_artifacts` | residual LaTeX that leaked into rendered text |
| `source_expects_math`, `source_expects_figures` | `NULL` means the source was never read — distinct from `False` |
| `failed_gates` | which quality gates rejected the document |
| `duration_s` | subprocess wall clock, always |

`source_expects_*` are graded from the same source string the quality gates saw,
so changing a source-dependent gate later is a Parquet-only operation instead of
a re-read of terabytes of source tars.

## Sizing

Roughly 44 core-seconds per document, so a full arXiv snapshot is on the order
of 79,000 core-hours.

`--papers-per-task` bounds peak memory and retry granularity. Conversion cost is
driven by HTML *density per document*, not document count: the densest shard by
row count is a 2003 shard at 3,840 rows, but the heaviest is a 2014 shard with
only 660 rows and 1.68 GB of HTML — 2.5 MB/doc. Sizing on the assumption that
the oldest shards are the big ones is backwards; they are numerous, not heavy.

Documents whose HTML exceeds 64 MB are rejected as `suspect_oversized` rather
than written. One paper measured 1,444 MB of HTML, 86% of its shard's entire
output, and drove peak RSS to 10.2 GB on its own.

`--timeout` defaults to 600s. Completion rate at 600s and at 2700s is identical
over the profiled sample: every document that finishes at all finishes well
inside 600s, and the rest are runaways that exhaust any budget and fail anyway.

## Figures

`--asset-dir` writes rasterized figures LaTeXML produced, one tar per shard,
named `<shard>-<partition>.tar` with members `<arxiv_id>/<file>`. Omitted by
default: most submissions rasterize nothing, and the HTML references figures by
name whether or not you keep the bytes.
