# Essential AI (EAI) Crawl — PDF URL Extraction

Pipeline for scanning the Essential AI targeted crawl (~3 PiB of WARCs) and
emitting PDF response URLs + metadata. The crawl is STEM / math / finance
focused and contains mixed MIME types (HTML, PDF, images, video, …); this
tutorial currently keeps only `Content-Type: application/pdf` responses.

## Dataset

| | |
|---|---|
| **Store** | SwiftStack / S3-compatible (`team-vendor-data` rclone remote) |
| **Bucket** | `vdi-169-essentialai-essentialai-data` |
| **Layout** | `eai-warc/<YYYYMMDD>/<uuid>.warc.gz` |
| **Scale** | ~3 PiB compressed |

```bash
# fish (login)
rclone lsd team-vendor-data:vdi-169-essentialai-essentialai-data/eai-warc | head
rclone lsf team-vendor-data:vdi-169-essentialai-essentialai-data/eai-warc/20240814 | head
```

Similar in spirit to Common Crawl, but **no public CDX / columnar index is
shipped with this dataset**. Random access into records therefore requires
building our own index (see [Indexing](#indexing-for-o1-pdf-extraction)).

## What this pipeline does

1. List WARC objects (local dir or S3 prefix).
2. Stream / scan each WARC; keep `response` records whose HTTP `Content-Type`
   contains `application/pdf`.
3. Emit Parquet rows with URL + light metadata (**PDF bodies are not kept**).

Output columns (`PDF_OUTPUT_COLUMNS` + stream index fields):

| Column | Meaning |
|--------|---------|
| `url` | `WARC-Target-URI` |
| `warc_id` | WARC-Record-ID (uuid; stripped of `<urn:uuid:…>`) |
| `content_type` | HTTP Content-Type (normalized) |
| `content_length` | HTTP/payload length when known (not the Range-GET size) |
| `http_status` | HTTP status code |
| `warc_date` | `WARC-Date` |
| `filename` | Basename derived from the **URL** path (e.g. `Medicion4.pdf`) |
| `warc_filename` | Full object key (`eai-warc/YYYYMMDD/<uuid>.warc.gz`) |
| `warc_record_offset` / `warc_record_length` | Compressed member bounds for O(1) Range-GET |
| `file_name` | Basename of `warc_filename` (convenience; kept intentionally) |

Dropped as duplicates: `id` (== `warc_id`), `source_id` (== `file_name`).

## Access modes

| Mode | Flag | When to use |
|------|------|-------------|
| **S3 streaming** | `--s3-bucket … --stream` | Default for this crawl (`.warc.gz`). Reads the whole object through `warcio`; bodies are discarded in-flight. |
| **S3 range metadata** | `--s3-bucket …` (no `--stream`) | Uncompressed `.warc` only — range-reads headers and skips bodies. |
| **Local / shared FS** | `--warc-dir …` | WARCs already on Lustre/NFS. |

Whole-file gzip cannot be range-read for individual records without an index;
that is why the production path for EAI is `--stream`.

## Run instructions

### Prerequisites

```bash
# fish (login)
cd /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
uv sync --extra text_cpu
# For the CDX probe / S3 streaming helpers:
uv sync --group eai-warcs
# S3/SwiftStack: AWS_* creds + endpoint (e.g. https://pdx.s8k.io); path-style addressing
```

### One-pass CDX + PDF probe (single object)

```bash
# fish (login)
uv run --group eai-warcs --no-sync python tutorials/eai_crawl/probe_cdx_index.py \
    --bucket vdi-169-essentialai-essentialai-data \
    --key eai-warc/20240814/01a4607c-0aa4-4159-b8dd-8b9f9c8af0da.warc.gz \
    --endpoint-url https://pdx.s8k.io \
    --output-dir /tmp/eai_cdx_probe
```

Verified on that sample: **per-record gzip (CC-style)**, 722 response CDX rows with
monotonic `(offset, length)` spanning the object — O(1) range-fetch is viable.
That particular shard had **0** `application/pdf` responses (mostly `text/html`).

### Find a PDF + O(1) range-fetch verification

```bash
# fish (login) — scans day prefix until one PDF, then Range-GETs only that member
uv run --group eai-warcs --no-sync python tutorials/eai_crawl/find_and_fetch_pdf.py \
    --bucket vdi-169-essentialai-essentialai-data \
    --prefix eai-warc/20240814/ \
    --endpoint-url https://pdx.s8k.io \
    --output-dir /tmp/eai_pdf_fetch \
    --skip-key 01a4607c-0aa4-4159-b8dd-8b9f9c8af0da.warc.gz
```

Verified: PDF at `https://yiyanglin.com/files/Intel_JD.pdf` in
`05f11429-….warc.gz` — fetched **7 312** compressed bytes (vs **10.5 MB** object)
via `bytes=6175422-6182733`, recovered a valid `%PDF-1.7` body (12 083 bytes).

### Local smoke test (single WARC, no Ray)

```bash
# fish (login)
uv run --extra text_cpu python tutorials/eai_crawl/run_local.py \
    --warc /path/to/sample.warc.gz \
    --output-dir /tmp/eai_pdf_urls
```

### Output layout (partition by day)

Prefer **Hive-style day partitions** so the full corpus is one table and days stay
independent / resumable:

| Path | Contents |
|------|----------|
| `s3://eai-warcs/pdf_url_idx/crawl_date=YYYYMMDD/warc_group=NNNNN/*.parquet` | PDF URLs + offsets |
| `s3://eai-warcs/cdx/crawl_date=YYYYMMDD/warc_group=NNNNN/<uuid>.parquet` | Full response CDX per WARC |

- **Stress-test one day:** write only `crawl_date=20240814/`.
- **Full crawl later:** one Slurm array per selected day; every element owns one
  `warc_group` subtree, so overwrite mode cannot delete a sibling's output.
- **Do not** dump all days into a flat `pdf_url_idx/` with no partition — harder to
  resume, reprocess, or delete a bad day. Readers (DuckDB/Spark/pandas) can still
  scan the whole tree as one dataset.

Smoke tests use a separate prefix (`…_smoke` or `crawl_date=…_smoke`) so they are
easy to delete without touching real day partitions.

### Day-scale checklist (you run these)

**Admission unit:** one day-prefix (`eai-warc/20240814/`) = one Slurm array.
The login-side launcher lists that day once and splits it into fixed-count key
files. Each array element receives one exact group; inside it, Curator fans out
to **one Ray task per WARC**.

Write remote: rclone ``eai-data`` (separate creds from the WARC read remote).

**1. Sync deps (login, once):**

```fish
# fish (login)
cd /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
uv sync --group eai-warcs --extra text_cpu
```

**2. Smoke (no Ray — preferred on shared login nodes):**

Shared login nodes often have other users' Ray clusters holding worker ports
(``Address already in use … :10002``). Use this sequential smoke instead of
``run_slurm.py`` on the login node; use SLURM for the full day.

```fish
# fish (login)
cd /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
uv sync --group eai-warcs --extra text_cpu
set -x PATH (pwd)/.venv/bin $PATH

set -x AWS_ACCESS_KEY_ID (rclone config show team-vendor-data | string match -r 'access_key_id = .*' | string replace 'access_key_id = ' '')
set -x AWS_SECRET_ACCESS_KEY (rclone config show team-vendor-data | string match -r 'secret_access_key = .*' | string replace 'secret_access_key = ' '')
set -x AWS_ENDPOINT_URL https://pdx.s8k.io
set -x AWS_DEFAULT_REGION us-east-1

python tutorials/eai_crawl/run_day_smoke.py \
    --s3-bucket vdi-169-essentialai-essentialai-data \
    --s3-prefix eai-warc/20240814/ \
    --s3-endpoint-url https://pdx.s8k.io \
    --url-limit 2 \
    --output-dir s3://eai-warcs/pdf_url_idx/crawl_date=20240814_smoke/ \
    --cdx-output-dir s3://eai-warcs/cdx/crawl_date=20240814_smoke/ \
    --output-rclone-remote eai-data
```

Verify:

```fish
rclone lsf --s3-force-path-style=true eai-data:eai-warcs/pdf_url_idx/crawl_date=20240814_smoke/
rclone lsf --s3-force-path-style=true eai-data:eai-warcs/cdx/crawl_date=20240814_smoke/
```

**Cleanup smoke garbage (run after verifying, before the real day job):**

```fish
# fish (login) — deletes ONLY *_smoke partitions / local scratch
rclone purge --s3-force-path-style=true eai-data:eai-warcs/pdf_url_idx/crawl_date=20240814_smoke
rclone purge --s3-force-path-style=true eai-data:eai-warcs/cdx/crawl_date=20240814_smoke
# older flat smoke prefixes (if any):
rclone purge --s3-force-path-style=true eai-data:eai-warcs/pdf_url_idx/20240814_smoke 2>/dev/null
rclone purge --s3-force-path-style=true eai-data:eai-warcs/cdx/20240814_smoke 2>/dev/null
rm -rf /lustre/fsw/portfolios/nemotron/users/tromanski/eai_out \
       /tmp/eai_cdx_probe /tmp/eai_pdf_fetch /tmp/eai_pdf_verify /tmp/eai_pdf_idx_smoke /tmp/eai_pdf_20240814
```

**3. Full day as a one-node-per-group Slurm array:**

Do **not** use `uv run` under Ray — `submit.sh` activates `.venv`. Preview the
exact worklists and `sbatch` command before submitting:

```bash
# login node; reuse the source AWS_* credentials from step 2
DRY_RUN=1 WARCS_PER_ARRAY_TASK=720 \
  bash tutorials/eai_crawl/run_day_array.sh 20240814

# Reuse the exact worklist path printed by the dry run; this does not re-list.
EXISTING_WORKLIST_DIR="$(pwd)/logs/eai_array_worklists/20240814.active" \
  bash tutorials/eai_crawl/run_day_array.sh 20240814
```

The default derives 720 WARCs from an initial estimate of 240 WARCs/hour for a
three-hour target, with a four-hour Slurm ceiling. This is not yet a remote
throughput measurement. Calibrate the next group size from a representative
one-node result (`new_count = old_count * 3h / elapsed`) before widening the
array. Submit only one day/campaign at a time unless the sum of array throttles
still respects the workflow-wide node cap. The `.active` worklist directory is
also the campaign claim and is retained as a receipt; do not remove it until
the submitted array has stopped.

Watch: `tail -f logs/eai_20240814_<array-job>_<task>.log`.

**4. After the day finishes — URL handoff:**

```fish
# fish (login)
rclone copy --s3-force-path-style=true \
  eai-data:eai-warcs/pdf_url_idx/crawl_date=20240814/ /tmp/eai_pdf_20240814/
uv run --group eai-warcs --no-sync python - <<'PY'
from pathlib import Path
import pandas as pd
pdf_root = Path("/tmp/eai_pdf_20240814")
pdfs = pd.concat([pd.read_parquet(p) for p in pdf_root.rglob("*.parquet")], ignore_index=True)
print("pdf rows", len(pdfs), "unique urls", pdfs["url"].nunique())
pdfs["url"].drop_duplicates().to_csv(pdf_root / "urls_for_robots.txt", index=False, header=False)
PY
```

## One-pass CDX sample (single S3 object)

Streams one `.warc.gz` from SwiftStack and writes both the CDX-style index and
PDF URL rows (no Ray required):

```fish
# fish (login) — uses rclone remote `team-vendor-data` for creds if AWS_* unset
cd /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
uv run --extra text_cpu python tutorials/eai_crawl/run_cdx_sample.py \
    --bucket vdi-169-essentialai-essentialai-data \
    --key eai-warc/20240814/01a4607c-0aa4-4159-b8dd-8b9f9c8af0da.warc.gz \
    --endpoint-url https://pdx.s8k.io \
    --output-dir /tmp/eai_cdx_sample
```

If boto3 hits virtual-hosted URL issues, stream via rclone instead:

```fish
uv run --extra text_cpu python tutorials/eai_crawl/run_cdx_sample.py \
    --via-rclone-cat \
    --output-dir /tmp/eai_cdx_sample
```

Outputs:

- `cdx.parquet` / `cdx_sample.csv` — all response records with
  `(url, warc_filename, warc_record_offset, warc_record_length, …)`
- `pdfs.parquet` / `pdfs.csv` — PDF rows plus the same offset/length fields

Core logic: `cdx_index.iterate_cdx_and_pdfs`.

## Robots.txt URL set (handoff)

After a pass completes, the Parquet `url` column is the set to send for
robots / allowlist filtering. Example:

```bash
# fish (login) — unique URLs (or domains) for the other team
uv run --extra text_cpu python - <<'PY'
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd

root = Path("/shared/out/eai_pdf")
urls = (
    pd.concat([pq.read_table(p, columns=["url"]).to_pandas() for p in root.rglob("*.parquet")])
    ["url"].dropna().drop_duplicates()
)
urls.to_csv("/shared/out/eai_pdf_urls.txt", index=False, header=False)
print(f"{len(urls)} unique URLs")
# Optional domain list:
# from urllib.parse import urlparse
# domains = sorted({urlparse(u).netloc.lower() for u in urls})
PY
```

Ship `eai_pdf_urls.txt` (or a domain list) to the filtering team. Keep the
Parquet (with `warc_id` / `warc_filename` + offsets) so filtered URLs can be joined back.

## Indexing for O(1) PDF extraction

### Can we build the index and collect PDF URLs in one pass?

**Yes.** A single streaming pass over each `.warc.gz` can emit:

1. **PDF URL / metadata rows** (what the pipeline writes today), and
2. **A CDX-style index** with enough fields for later random access:

| Field | Role |
|-------|------|
| `url` | Join key after robots filtering |
| `warc_filename` | Object key under `eai-warc/…` |
| `warc_record_offset` | Byte offset of the record in the object |
| `warc_record_length` | Compressed (or raw) record length |
| `content_mime_type`, `http_status`, `warc_id`, … | Optional filters / provenance |

`warcio.ArchiveIterator` exposes `get_record_offset()` / `get_record_length()`
while iterating; record those alongside the PDF filter. No second full scan is
required for indexing + URL extraction.

You can either:

- **PDF-only index** — smaller; enough if you only ever re-fetch PDFs, or
- **Full CDX** — every `response` (or every record); more like Common Crawl’s
  columnar index, reusable for other MIME types later.

### O(1) fetch caveat: gzip layout

Common Crawl’s O(1) path works because `.warc.gz` files are **per-record
gzip** (each record is its own gzip member). Given `(offset, length)` you
S3/HTTP Range-GET that slice, gunzip it, and parse one record.

| Layout | Index useful for range-GET? |
|--------|-----------------------------|
| **Per-record gzip** (CC-style) | Yes — true O(1) fetch of filtered PDFs |
| **Whole-file gzip** (one member) | Offsets into the compressed object are **not** independently seekable; range-GET cannot reconstruct a mid-file record |

**Before investing in a full 3 PiB index pass, verify the EAI layout** on a
sample object (count gzip members / compare warcio offsets). If it is
whole-file gzip, options are:

1. Index **uncompressed** offsets and materialize/gunzip to `.warc` (or a
   seekable cache) before range reads, or
2. Re-encode to per-record gzip while building the index (expensive rewrite), or
3. Keep streaming and accept O(n) per shard when extracting a filtered subset
   (still fine if you only re-read shards that contain kept URLs).

Code already prepared for the CC-style path once offsets exist:
`WarcMetadataScanner.read_record_metadata` in `s3_download.py`.

### Suggested end-to-end flow

```
Pass 1 (this tutorial, extended):
  stream each eai-warc/<day>/*.warc.gz
    → Parquet: PDF urls + metadata
    → Parquet/CDX: (url, warc_filename, offset, length, …)

Handoff:
  unique urls (or domains) → robots / policy team

Pass 2 (after allowlist returns):
  join allowlisted urls ⋈ index
    → (warc_filename, offset, length)
    → range-fetch PDF bodies (if per-record gzip) or targeted re-stream
```

## Layout of this tutorial

| File | Role |
|------|------|
| `pdf_records.py` | Shared PDF filter + column schema |
| `s3_streaming.py` | Production path for compressed EAI `.warc.gz` |
| `s3_stage.py` / `s3_download.py` | Uncompressed range-scan + CDX-offset reader |
| `stage.py` | Local-FS download/iterate composite |
| `run_local.py` | No-Ray smoke test |
| `run_slurm.py` | Ray / SLURM entrypoint |
| `submit.sh` | Multi-node `sbatch` wrapper |

## Related

- Common Crawl index + byte-range fetch pattern: `tutorials/math/README.md`
- `CommonCrawlWARCReader` (offset/length → Range GET):
  `nemo_curator/stages/text/download/common_crawl/download.py`
