# arXiv LaTeX → Interleaved Parquet

Turn raw arXiv source tars into **interleaved multimodal rows** — body text, figure, caption, body text — in the document's own reading order, ready for the existing interleaved filters and writers.

Every other arXiv path in Curator ([`ArxivDownloadExtractStage`](../../../nemo_curator/stages/text/download/arxiv/)) throws the figures away and emits one plain-text blob per paper. This pipeline keeps them, and keeps the position where they belong in the prose.

## What the Pipeline Does

```
s3://arxiv/src/arXiv_src_0001_001.tar        one shard, ~2364 submissions, ~500 MB
        │
        ▼
FilePartitioningStage                        group shard paths into tasks
        │
        ▼
ArxivTarPartitioningStage                    read only the tar's member headers
        │                                    (~0.35 s for a 227 MB shard — tarfile
        │                                    seeks, it does not read) and hand out
        │                                    byte ranges, papers_per_task at a time
        ▼
ArxivLatexReaderStage                        seek → gunzip → inner tar → parse LaTeX
        │                                    → one row per document element
        ▼
InterleavedBatch  ──▶  optional filter  ──▶  InterleavedParquetWriterStage
```

`ArxivLatexReader` is the `CompositeStage` that wires the first three together; you add one stage, not three.

Inside the reader, each submission is a gzip member that expands to either a tar of the project's files or a single bare `.tex` file. In the January 2000 shard the split is 1631 / 702 out of 2364, plus 31 bare PDFs with no source at all, which are counted and skipped. The parser then:

1. Picks the root `.tex` (the one with `\documentclass` or `\documentstyle` and a `\begin{document}`).
2. Inlines `\input` / `\include` files, expands simple user macros, and strips `%` comments.
3. Walks the body, splitting it at every float and emitting text runs, figures, and captions in order.
4. Resolves each graphics reference against the project's files, honoring `\graphicspath`.

Graphics resolution is deliberately broad because 2000-era papers barely use `\includegraphics`. The parser handles `\epsfxsize`/`\epsfbox`/`\epsffile`, `\epsfig`, `\psfig`, `\plotone`/`\plottwo`/`\plotfiddle`, and `\special{psfile=...}` alongside it.

## Get the Data

arXiv publishes bulk source dumps as `arXiv_src_YYMM_NNN.tar` in the **requester-pays** bucket `s3://arxiv/src/`. You pay for listing and transfer, so start with one shard.

Requirements: an AWS account with working credentials and [`s5cmd`](https://github.com/peak/s5cmd).

```bash
pip install s5cmd

# List what is there (this is what ArxivUrlGenerator does internally --
# see nemo_curator/stages/text/download/arxiv/url_generation.py).
s5cmd --request-payer=requester ls 's3://arxiv/src/' | grep '.tar'

# Pull a single shard (~500 MB) to work with locally.
mkdir -p /data/arxiv-src
s5cmd --request-payer=requester cp 's3://arxiv/src/arXiv_src_0001_001.tar' /data/arxiv-src/

# Or a whole month.
s5cmd --request-payer=requester cp 's3://arxiv/src/arXiv_src_0001_*.tar' /data/arxiv-src/
```

With `aws s3` instead, the equivalent flag is `--request-payer requester`.

You can also point the reader straight at S3 and skip the local copy — see the `--storage-options-json` example below — but for repeated runs a local copy is cheaper.

## Install

```bash
git clone https://github.com/NVIDIA-NeMo/Curator.git
cd Curator
pip install uv
uv sync --extra interleaved_cpu     # or --extra interleaved_cuda12
```

Nothing in this pipeline needs a GPU: it is tar seeking, gunzip, and regex.

## Run It

```bash
python tutorials/interleaved/arxiv_regex/run.py \
    --input-dir /data/arxiv-src/ \
    --output-dir /data/arxiv-interleaved/ \
    --mode overwrite
```

`run.py` starts a local Ray cluster via `RayClient()`. **Ray is required** — the pipeline executor schedules every stage as Ray tasks, so this will not run on a login node or anywhere process/memory limits are tight. To poke at a single stage without Ray, build it directly and call `stage.process(task)`.

### Useful Flags

| Flag | Default | What it does |
|---|---|---|
| `--input-dir` | required | Directory, path, or glob of `arXiv_src_*.tar`. Local or `s3://`. |
| `--output-dir` | required | Where Parquet files land. |
| `--papers-per-task` | `100` | Submissions handled per reader task. This is the memory dial: a task holds that many decompressed projects and their figure bytes. |
| `--max-papers-per-tar` | all | Index at most N submissions per shard. Use it for smoke tests. |
| `--text-only` | off | `include_images=False`. No figure bytes at all — much faster, output shrinks by ~an order of magnitude. Captions are still emitted. |
| `--min-text-chars` | `0` | Enables `ShortTextFilterStage`, which drops body-text runs shorter than N characters. Captions are exempt so figure/caption pairs stay intact. |
| `--image-content-types` | all | Keep only figures whose sniffed MIME type matches, e.g. `image/png image/jpeg`. |
| `--no-captions` | off | Do not emit caption rows. |
| `--keep-bibliography` / `--drop-appendix` | off / off | Where to truncate the body. |
| `--max-batch-bytes` | `None` | Split reader output into roughly this many bytes per batch, never splitting a paper. |
| `--storage-options-json` | `None` | fsspec options for the input, e.g. `'{"requester_pays": true}'`. |

Smoke test on 200 papers per shard:

```bash
python tutorials/interleaved/arxiv_regex/run.py \
    --input-dir /data/arxiv-src/ \
    --output-dir /tmp/arxiv_smoke/ \
    --max-papers-per-tar 200 \
    --papers-per-task 50 \
    --mode overwrite
```

Text-only pass with a length filter:

```bash
python tutorials/interleaved/arxiv_regex/run.py \
    --input-dir /data/arxiv-src/ \
    --output-dir /data/arxiv-text/ \
    --text-only \
    --min-text-chars 200 \
    --mode overwrite
```

Read the bucket directly:

```bash
python tutorials/interleaved/arxiv_regex/run.py \
    --input-dir s3://arxiv/src/ \
    --output-dir /data/arxiv-interleaved/ \
    --storage-options-json '{"requester_pays": true}' \
    --mode overwrite
```

### From Python

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.latex.arxiv.regex import ArxivLatexReader

pipeline = Pipeline(name="arxiv_latex")
pipeline.add_stage(ArxivLatexReader(file_paths="/data/arxiv-src/", papers_per_task=100))
pipeline.add_stage(InterleavedParquetWriterStage(path="/data/arxiv-interleaved/", mode="overwrite"))
pipeline.run()
```

## Output Schema

Rows follow the standard `INTERLEAVED_SCHEMA` (see [`nemo_curator/stages/interleaved/README.md`](../../../nemo_curator/stages/interleaved/README.md)) plus five arXiv-specific columns.

### Reserved columns, as this reader fills them

| Column | Value |
|---|---|
| `sample_id` | The canonical arXiv id (`astro-ph/0001001`, `2401.00001`). One sample per paper. |
| `position` | `-1` for the metadata row, then `0..n` in reading order. |
| `modality` | `metadata`, `text`, or `image`. |
| `content_type` | `application/json` for metadata, `text/x-tex` for text and captions, sniffed MIME type for figures. |
| `text_content` | Text/caption payload, or the JSON provenance blob on the metadata row. |
| `binary_content` | Figure bytes, materialized eagerly (see below). |
| `source_ref` | Locator for the **submission**: shard path, member name, byte offset, byte size. |
| `materialize_error` | Set only when `on_missing_graphics="annotate"` and a graphics file could not be found. |

### The five extra columns

| Column | Type | Description |
|---|---|---|
| `arxiv_id` | string | Canonical arXiv id, repeated on every row of the paper. Same value as `sample_id`, but survives any downstream re-keying of `sample_id`. |
| `element_class` | string | Finer-grained than `modality`: `metadata`, `text`, `figure`, or `caption`. This is the column that separates a body-text run from a caption — both are `modality="text"`. |
| `graphics_file` | string | The project-relative file the figure was resolved to, e.g. `figures/spectrum.eps`. Null for non-figure rows. |
| `figure_id` | int64 | Per-paper figure counter, `0..n`. Shared by a figure row and its caption row, so you can join them. Null on a caption whose figure was dropped. |
| `figure_label` | string | The float's `\label{...}`, when it has one — this is what `\ref{fig:spectrum}` in the body points at. |

`element_class` and `figure_id` together are what make the output usable: `WHERE element_class = 'caption' AND figure_id = 3` gets you the caption of figure 3.

### Row layout for one paper

```
position  modality   element_class  content_type              content
      -1  metadata   metadata       application/json          {"arxiv_id": ..., "root_tex": ..., "num_unresolved_graphics": 0, ...}
       0  text       text           text/x-tex                "We report on observations of ..."
       1  image      figure         application/postscript    <bytes>          figure_id=0  graphics_file=f1.eps
       2  text       caption        text/x-tex                "Fig. 1. Spectrum of ..."   figure_id=0
       3  text       text           text/x-tex                "The spectrum shows ..."
       ...
```

The metadata row's JSON carries `arxiv_id`, `source_tar`, `source_member`, `root_tex`, `num_text_rows`, `num_image_rows`, `num_unresolved_graphics`, and `graphics_paths` — enough to audit a paper without re-reading the shard.

Multi-panel floats (`\plottwo`, several `\epsfbox` calls in one `figure` environment) emit one image row per panel but **one** caption row, after the last panel, so the caption text is not duplicated N times.

### Why figure bytes are eager

A figure lives inside an inner tar, inside a gzip member, inside the shard. That nesting is not addressable by the `source_ref` locator the interleaved materialization helpers understand, so the reader materializes figure bytes at read time. `source_ref` therefore points at the *submission*, which is both resolvable and the right granularity for provenance and dedup. Consequence: set `materialize_on_write=False` on the writer (`run.py` does), and use `--papers-per-task` to bound worker memory.

## Measured Quality

Numbers below come from full-shard scans of `arXiv_src_0001_001.tar` (January 2000) and an end-to-end run over 300 papers from it.

**Shard composition** — 2364 submissions: ~1631 inner-tar projects, ~702 single-file `.gz`, ~31 bare `.pdf` (skipped, no source).

**Why the parser is not just `\includegraphics`** — graphics macros by paper count in that shard:

| Macro | Papers | | Macro | Papers |
|---|---:|---|---|---:|
| `\epsfxsize` | 453 | | `\plotone` | 132 |
| `\epsfbox` | 398 | | `\plotfiddle` | 86 |
| `\epsfig` | 347 | | `\plottwo` | 81 |
| `\psfig` | 318 | | `\special{psfile=}` | 73 |
| `\epsffile` | 197 | | **`\includegraphics`** | **224 (9.6%)** |

1624 papers still use `\documentstyle` (LaTeX 2.09). Handling only the modern macro would miss most of the era's figures.

**Two details that moved the numbers most:**

- Stripping `%` comments before scanning for graphics raised resolution from **86.6% → 95.6%**. Commented-out `\psfig` lines were resolving to files that the document never actually includes.
- **58.9%** of `\caption` bodies contain nested braces, so caption extraction has to brace-match rather than regex to the first `}`.

**End-to-end on 300 papers:**

| Metric | Result |
|---|---|
| Unresolved graphics references | **0.2%** |
| Figures with a caption attached | **93%** |
| Papers producing empty text | **0** |
| Throughput | **~52 papers/s**, single-threaded |

Scale the throughput number with care: it is one core on local NVMe, with figure bytes included. Reading from S3 is network-bound instead.

## Limitations — Read This Before You Train On It

**Figures are ~96% PostScript.** In this era `.eps`/`.ps` is the dominant figure format, and no browser and few image libraries render it. The bytes are preserved faithfully, but if your consumer expects PNG/JPEG you must either:

- rasterize downstream (Ghostscript, `epstopdf` + `pdftoppm`), or
- drop non-raster figures at read time with `--image-content-types image/png image/jpeg`, which on early shards leaves you with very few figures.

This gets steadily better in later years; run the numbers on the shards you actually plan to use rather than assuming the 2000 ratio holds.

**Text is still LaTeX.** `content_type` is `text/x-tex`, not `text/plain`. Cleaning removes comments, in-body macro definitions, and layout-only commands, but math, citations, and semantic markup are left intact on purpose — detexing is lossy and belongs downstream where you can choose the policy. If you want plain text, add a stage; do not assume this one gave you any.

**Root-file selection can be wrong.** A project with several plausible root `.tex` files (paper + response letter + poster) may parse the wrong one. `root_tex` in the metadata row tells you which was chosen.

**Missing graphics are dropped by default.** `on_missing_graphics="skip"` silently drops the ~0.2% of references that do not resolve. Pass `on_missing_graphics="annotate"` to the reader to get a row with `materialize_error` set instead — but note the default writer policy turns that into a hard error, so pair it with `--on-materialize-error warn`.

## Related

- [`nemo_curator/stages/interleaved/README.md`](../../../nemo_curator/stages/interleaved/README.md) — the interleaved schema, materialization, and IO stages.
- [Getting Started tutorial](../getting-started/) — the same interleaved plumbing on MINT-1T WebDataset shards.
- [`nemo_curator/stages/text/download/arxiv/`](../../../nemo_curator/stages/text/download/arxiv/) — the text-only arXiv pipeline, if you do not need figures.
