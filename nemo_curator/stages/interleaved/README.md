# Interleaved Pipeline

Row-wise interleaved multimodal ingestion and write path for WebDataset tar shards (MINT-1T style), with materialization support for local, remote, and tar-archived binary content.

## Architecture

```
WebDataset tar shards          Parquet files
        |                            |
        v                            v
┌──────────────────────────┐  ┌──────────────────────────┐
│ InterleavedWebdataset-   │  │ InterleavedParquetReader  │  Both are CompositeStages:
│ Reader (io/reader.py)    │  │ (io/reader.py)            │  FilePartitioningStage +
│                          │  │                           │  <modality>ReaderStage
└──────────┬───────────────┘  └────────────┬─────────────┘
           └──────────┬────────────────────┘
                      |  InterleavedBatch (Arrow/Pandas)
                      v
         ┌─────────────────────────┐
         │  Filter Stages          │  e.g. InterleavedAspectRatioFilterStage
         │  (stages.py)            │  Row-wise filtering with optional materialization
         └────────┬────────────────┘
                  |
        ┌─────────┴──────────┐
        v                    v
┌───────────────┐   ┌──────────────────────────┐
│ Interleaved-  │   │ InterleavedWebdataset-    │
│ ParquetWriter │   │ WriterStage               │
│ Stage         │   │ (io/writers/webdataset.py)│
│ (tabular.py)  │   │ MINT-1T-style tar shards  │
└───────────────┘   └──────────────────────────┘
```

## Schema (`INTERLEAVED_SCHEMA`)

Defined in `nemo_curator/tasks/interleaved.py`. Columns are split into **reserved** (managed by the pipeline) and **user** (passthrough from source data).

### Reserved columns (`RESERVED_COLUMNS`)

These are set and managed by pipeline stages. Users should not write to them directly.

| Column | Type | Category | Description |
|--------|------|----------|-------------|
| `sample_id` | string (required) | Identity | Unique document/sample identifier |
| `position` | int32 (required) | Identity | Position within sample (-1 for metadata rows) |
| `modality` | string (required) | Identity | Row modality: `text`, `image`, `metadata` built-in; extensible to `audio`, `table`, `generated_image`, etc. |
| `content_type` | string | Content | MIME type (e.g. `text/plain`, `image/jpeg`) |
| `text_content` | string | Content | Text payload for text rows |
| `binary_content` | large_binary | Content | Image bytes (populated by materialization) |
| `source_ref` | string | Internal | JSON locator `{path, member, byte_offset, byte_size, frame_index}`. `path` alone = direct/remote read; + `member` = tar extract; + `byte_offset/size` = range read (fastest). `path` accepts local or remote (`s3://`) URIs. |
| `materialize_error` | string | Internal | Error message if materialization failed |

### User columns (passthrough)

Extra fields from the source data flow through the pipeline as additional columns. Specify them with the `fields` parameter on the reader:

```python
reader = InterleavedWebdatasetReader(
    file_paths="/data/shards/",
    fields=("p_hash", "score", "aux"),  # These become extra columns
)
```

If `fields` is `None` (default), all non-reserved fields from the source JSON are passed through. If specified explicitly, only the listed fields are included -- and the reader validates they exist and don't collide with reserved names.

## Key Concepts

### InterleavedBatch

The task type for interleaved multimodal data (`nemo_curator/tasks/interleaved.py`). Wraps either a PyArrow Table or Pandas DataFrame.

Class attributes:
- `REQUIRED_COLUMNS` -- frozenset of columns that must always be present (non-nullable schema fields)

Key methods:
- `build_source_ref(path, member, byte_offset, byte_size, frame_index)` -- build a JSON locator string
- `parse_source_ref(value)` -- parse back with soft migration for older formats
- `with_parsed_source_ref_columns(prefix)` -- expand source_ref into DataFrame columns
- `to_pyarrow()` / `to_pandas()` -- conversion between formats

### source_ref

A JSON string embedded in each row that tracks where the original content lives:

```json
{
  "path": "/data/shard-00000.tar",
  "member": "abc123.jpg",
  "byte_offset": 1024,
  "byte_size": 45678,
  "frame_index": null
}
```

- `path` + `member` -- tar archive path and member name
- `path` alone (no member) -- direct file path
- `byte_offset` + `byte_size` -- enables range reads without opening the tar
- `frame_index` (optional) -- selects a single frame from a multi-frame TIFF during materialization

### Materialization

Binary content (images) can be loaded lazily. Three I/O strategies dispatch automatically based on `source_ref` content (`utils/materialization.py`):

| Strategy | When | How |
|----------|------|-----|
| **Range read** | `byte_offset` + `byte_size` present | `fs.cat_ranges()` -- batched HTTP range requests per path |
| **Tar extract** | `member` present, no byte range | Open tar once, `extractfile()` per member |
| **Direct read** | No `member` | Read entire file via `fsspec.open()` |

When `frame_index` is set in the `source_ref`, materialization extracts a single frame from a multi-frame TIFF and returns it as a standalone TIFF. Non-TIFF content is returned unchanged regardless of `frame_index`.

Materialization can happen at read time (`materialize_on_read=True`) or write time (`materialize_on_write=True`).

## Usage

```python
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io import InterleavedWebdatasetReader, InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.stages import InterleavedAspectRatioFilterStage

pipeline = Pipeline(name="mint1t_pipeline")
pipeline.add_stage(InterleavedWebdatasetReader(
    file_paths="/data/mint1t/shards/",
))
pipeline.add_stage(InterleavedAspectRatioFilterStage(drop_invalid_rows=True))
pipeline.add_stage(InterleavedParquetWriterStage(
    path="/output/parquet/",
    materialize_on_write=True,
    mode="overwrite",
))
pipeline.run()
```

## LaTeX Ingestion (arXiv)

Two paths convert arXiv bulk source tarballs, in separate subpackages under
`latex/arxiv/`. The package re-exports neither, so an import names the path it
means.

### `arxiv.latexml`

Runs `latexmlc`, a full LaTeX processor, over each submission. Output is HTML5
with presentation MathML: math is rendered as MathML with the original TeX kept
in `alttext`, and macros are expanded by the TeX engine. Emits one row per
submission -- including submissions that produced no HTML, so the denominator
stays "of all submissions" -- with the grading inputs retained so a change to a
source-dependent gate can be re-evaluated from Parquet.

Each document costs roughly 44 core-seconds.

```python
from nemo_curator.stages.interleaved.latex.arxiv.latexml.stage import ArxivLatexmlReader

pipeline.add_stage(ArxivLatexmlReader(file_paths="/data/arxiv-src/", papers_per_task=100))
```

To convert a single project without a pipeline:

```python
from nemo_curator.stages.interleaved.latex.arxiv.latexml.document import convert_submission

doc = convert_submission(payload, member, shard, digest, workdir)
if doc.converted:
    ...  # doc.html, doc.tier, doc.status
```

Requires the `latexmlc` binary on `PATH`. It is a Perl binary, not a Python
package, so it cannot be installed with pip. Obtain it from the public ar5iv
image, which bundles LaTeXML with the ar5iv bindings this path expects:

```bash
docker pull latexml/ar5ivist:2512.17     # amd64 only; no aarch64 build exists
```

Run the pipeline inside that image, or point `LatexmlConfig(executable=...)` at
your own build. `LatexmlConvertStage.setup()` checks for the binary once per
worker, so a missing converter fails the run immediately rather than once per
document.

### `arxiv.regex`

Pattern-matches the LaTeX source directly, without invoking a LaTeX processor.
Output is an `InterleavedBatch` whose rows follow the document's reading order
-- body text, figure, caption, body text -- ready for the interleaved filters
and writers described above. Math and unrecognised macros become placeholders,
and common macro definitions are expanded.

Each document costs milliseconds, and the path requires nothing beyond Curator.

```python
from nemo_curator.stages.interleaved.latex.arxiv.regex import ArxivLatexReader

pipeline.add_stage(ArxivLatexReader(file_paths="/data/arxiv-src/", papers_per_task=100))
```

Figures are read as stored. Pre-2005 submissions are dominated by PostScript, so
pass `image_content_types=("image/png", "image/jpeg")` to keep only figures that
are already raster, or handle conversion downstream.

## File Layout

```
stages/interleaved/
├── __init__.py                     # Exports filter/annotator stages
├── stages.py                       # BaseInterleavedAnnotatorStage, BaseInterleavedFilterStage,
│                                   # InterleavedAspectRatioFilterStage
├── io/
│   ├── __init__.py                 # Exports InterleavedWebdatasetReader, InterleavedParquetReader,
│   │                               # InterleavedParquetWriterStage, InterleavedWebdatasetWriterStage
│   ├── reader.py                   # InterleavedWebdatasetReader, InterleavedParquetReader (CompositeStages)
│   ├── readers/
│   │   ├── base.py                 # BaseInterleavedReader
│   │   ├── parquet.py              # InterleavedParquetReaderStage (ProcessingStage)
│   │   └── webdataset.py           # InterleavedWebdatasetReaderStage (ProcessingStage)
│   └── writers/
│       ├── base.py                 # BaseInterleavedWriter (filesystem + materialization + process)
│       ├── tabular.py              # InterleavedParquetWriterStage
│       └── webdataset.py           # InterleavedWebdatasetWriterStage
├── latex/
│   └── arxiv/                      # Two independent arXiv conversion paths
│       ├── __init__.py             # Describes both; re-exports neither
│       ├── latexml/                # Runs latexmlc -> HTML5 + presentation MathML
│       │   ├── model.py            # The types a conversion deals in
│       │   ├── convert.py          # LatexmlConfig, convert() -> ConversionResult
│       │   ├── document.py         # convert_submission(): one submission in, one out
│       │   ├── stage.py            # LatexmlConvertStage, ArxivLatexmlReader, HTML_SCHEMA
│       │   ├── extract.py          # Pick the root .tex out of a submission
│       │   ├── quality.py          # Tier/Status assessment of converted HTML
│       │   ├── artifacts.py        # Scan output for unresolved TeX artifacts
│       │   ├── source_text.py      # decode_text, strip_comments (own copy)
│       │   └── ...                 # boilerplate, profiling, runs, sampling
│       └── regex/                  # Pattern-matches source -> InterleavedBatch
│           ├── composite.py        # ArxivLatexReader (CompositeStage)
│           ├── reader.py           # ArxivLatexReaderStage
│           ├── partitioning.py     # ArxivTarPartitioningStage
│           ├── parsing.py          # parse_project, Figure, TextSegment
│           └── detex.py            # LaTeX -> plain text
└── utils/
    ├── constants.py                # Default file extensions
    ├── materialization.py          # Three-strategy materialization dispatch
    └── validation_utils.py         # Field validation, storage options resolution
```
