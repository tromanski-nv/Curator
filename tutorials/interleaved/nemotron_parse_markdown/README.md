# PDFs to interleaved markdown, in two phases

Nemotron-Parse tells you what is on each page. It does not tell you what the
document *says*: a sentence is cut in half by a column break, a figure sits in
the middle of a paragraph, the page number is a box like any other, and the
bibliography reads as prose. This pipeline runs the model and then applies the
rules that turn the one into the other, writing the result as markdown in the
interleaved schema.

```
PDFs ──▶ [ phase 1: parse ] ──▶ Nemotron-Parse elements ──▶ [ phase 2: post-process ] ──▶ interleaved markdown
             GPU, hours                 (parquet)                   CPU, minutes                (parquet)
```

The split is the point. Phase 1 costs a GPU-hour per few hundred documents and
its output is worth keeping. Phase 2 is pure Python over text, and it is the
half whose rules you actually want to tune. Run phase 1 once; re-run phase 2 as
often as you like.

## Setup

```bash
pip install uv
uv sync --extra interleaved_cuda12
```

## Quickstart

**Step 1 — a manifest listing your PDFs:**

```bash
for f in /path/to/pdfs/*.pdf; do
    echo "{\"file_name\": \"$(basename "$f")\"}" >> manifest.jsonl
done
```

**Step 2 — both phases at once:**

```bash
python tutorials/interleaved/nemotron_parse_markdown/run.py \
    --phase all \
    --manifest manifest.jsonl --pdf-dir /path/to/pdfs \
    --output-dir /path/to/markdown \
    --backend vllm --enforce-eager --max-pdfs 3
```

**Or, the way you will actually want it — parse once, iterate on the rules:**

```bash
# Once, on a GPU node.
python .../run.py --phase parse \
    --manifest manifest.jsonl --pdf-dir /path/to/pdfs \
    --output-dir /data/elements --backend vllm --enforce-eager

# As often as you like, anywhere.
python .../run.py --phase postprocess \
    --input-dir /data/elements --output-dir /data/markdown-v2 \
    --min-caption-chars 60 --no-skip-toc-bib
```

## What each phase does

### Phase 1 — parse

`NemotronParsePDFReader` decomposes into four stages:

| Stage | What it does |
|---|---|
| `pdf_partitioning` | Read the manifest, pack PDFs into tasks |
| `pdf_preprocess` | Extract PDF bytes, render pages to images |
| `nemotron_parse_inference` | Run the model (GPU) |
| `nemotron_parse_postprocess` | Decode the raw output into element rows, crop pictures |

Its output is the **Nemotron-Parse element format**: the interleaved schema,
one row per element, with `element_class`, `page_number` and a `source_ref`
holding `{page, bbox}`.

### Phase 2 — post-process

`NemotronParseMarkdownPostprocessor` decomposes into six stages, one per rule:

| Stage | What it does |
|---|---|
| `nemotron_parse_clean` | Condemn what never had content: empty, a Table that lost its `tabular`, a block degenerate with one repeated word |
| `nemotron_parse_assign_floats` | Lift tables, pictures, captions and footnotes out of the flow; match each caption to the figure nearest it |
| `nemotron_parse_page_furniture` | Condemn running heads and page numbers |
| `nemotron_parse_skip_sections` | Condemn the contents and the bibliography, from their heading until prose resumes |
| `nemotron_parse_reconstitute_paragraphs` | Rejoin a sentence broken across a column or a page |
| `nemotron_parse_markdown` | Flow before floats, written as markdown |

Six stages rather than one so a run can be stopped after any of them and the
intermediate written out — which is how you find out *which* rule dropped a
paragraph rather than that something did. `--fuse` collapses them into one
stage for throughput; the output is identical.

Nothing is deleted. A box that would not reach training is marked `keep=False`
with a reason, and `--emit-dropped` keeps those rows in the output so a viewer
can show the whole stream.

## What comes out

The interleaved schema (`sample_id`, `position`, `modality`, `content_type`,
`text_content`, `binary_content`, `source_ref`, `materialize_error`) plus
`element_class`, `page_number`, `source_positions`, `matched_to`, and whatever
the parse phase carried along. One document, abridged:

| position | modality | element_class | content_type | text_content |
|---|---|---|---|---|
| -1 | metadata | | application/json | `{"url": …, "pdf_name": …, "postprocess": {…stats…}}` |
| 0 | text | Title | text/markdown | `# Casimir Forces on a Quantum Emitter` |
| 1 | text | Section-header | text/markdown | `## I. INTRODUCTION` |
| 2 | text | Text | text/markdown | `We propose a model for a quantum emitter interacting with a dispersive object, via $\hat{H}_a$.` |
| 3 | text | Formula | text/markdown | `$$E_n = \hbar\omega(n + \tfrac{1}{2})$$` |
| 4 | image | Picture | image/png | *(bytes in `binary_content`)* |
| 5 | text | Caption | text/markdown | `FIG. 1. The emitter sits a distance d above…` |
| 6 | table | Table | text/markdown | `\begin{tabular}{cc}…\end{tabular}` |

Row 2 came from two elements — the two halves of a sentence broken by a column
break — and says so in `source_positions`. Row 5 records which figure it was
matched to in `matched_to`, as a `position` in this table. A paragraph rejoined
across a *page* boundary reports its `bbox` as null: it keeps the page it
started on, and the only box available describes the other page. Tables are left as LaTeX: there is no lossless
markdown pipe-table for `\multicolumn`/`\multirow`, which 47% and 23% of them
use. Inline math gets `$…$`; a `Formula` element, which occupies its own block,
gets `$$…$$`.

## Tuning the rules

Every rule can be switched off independently, which is what makes a
before/after comparison mean anything.

| Flag | Effect |
|---|---|
| `--no-assign-floats` | Leave figures and captions in the reading flow |
| `--no-reconstitute-paragraphs` | Do not rejoin broken sentences |
| `--no-skip-toc-bib` | Keep the contents and the bibliography |
| `--keep-page-furniture` | Keep running heads and page numbers |
| `--no-require-tabular` | Keep Tables with no `tabular` environment |
| `--no-check-repeated-words` | Keep degenerate repeated blocks |
| `--min-caption-chars`, `--min-caption-words` | What counts as a caption rather than layout noise |
| `--non-bib-toc-words` | How much prose ends a bibliography skip |
| `--strip-markdown` | Reduce markdown to plain text first (the predecessor pipeline's baseline) |
| `--text-only` | Drop pictures instead of interleaving them |
| `--drop-class Footnote` | Drop an element class outright (repeatable) |
| `--emit-dropped` | Keep condemned rows, marked with `keep=False` and `drop_reason` |

## Swapping the model version

Everything that differs between Nemotron-Parse releases lives in
`nemo_curator/stages/interleaved/pdf/nemotron_parse/versions.py`: which weights
to load, what prompt to send, and whether the model emits floats in reading
order. Moving to a later release that keeps the same output contract is one
argument:

```bash
python .../run.py --phase parse \
    --parse-version v2.0 --model-path nvidia/NVIDIA-Nemotron-Parse-v2.0 \
    ...
```

An unregistered version is accepted on your word that the contract is
unchanged, and says so in the log — but it needs `--model-path`, because
guessing a HuggingFace id is worse than asking for one. A *registered* release
names its own weights, so `--parse-version` alone is enough for it.

If a release changes the contract, describe what it actually does:

```python
from nemo_curator.stages.interleaved.pdf.nemotron_parse import (
    NemotronParseProfile, register_profile,
)

register_profile(
    NemotronParseProfile(
        name="v2.0",
        model_path="nvidia/NVIDIA-Nemotron-Parse-v2.0",
        markers=("v2.0",),
        floats_in_reading_order=True,
    )
)
```

Then `--parse-version v2.0` on its own picks up those weights and that
behaviour.

Phase 2 is unaffected either way: it reads element rows, not model output.

## Provenance

Pass `--atlas-id` and `--atlas-parent` and the run writes a `.atlas.json` into
its output directory recording what it produced, what it came from, and the
command verbatim — written by the run that made the data, which is the only
version of that worth anything.

```bash
python .../run.py --phase parse ... \
    --output-dir /data/elements \
    --atlas-id arxiv/nemotron-parse-elements/2026-08-21 \
    --atlas-parent arxiv/pdf/2026-07-27

python .../run.py --phase postprocess --input-dir /data/elements ... \
    --output-dir /data/markdown \
    --atlas-id arxiv/pdf-markdown/2026-08-21 \
    --atlas-parent arxiv/nemotron-parse-elements/2026-08-21
```

The elements are the canonical retained artifact; the markdown is derived and
gets its own id referencing it.

## A constraint worth knowing

A document must arrive whole: every row of one `sample_id` in one batch.
Paragraph reconstitution reaches across pages, so a document split between two
batches would come out split. The parse phase emits a document's rows together
and the Parquet writer keeps a task's rows in one file, so phase 2 reads its
input with `files_per_partition=1`.
