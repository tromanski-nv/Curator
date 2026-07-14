# Interleaved PDF deduplication

This workflow preserves identification artifacts and materializes a new Parquet dataset after each removal pass:

1. select the latest arXiv version represented in the frozen PDF snapshot;
2. identify byte-identical PDFs with SHA-256 and remove complete samples;
3. identify character-for-character exact extracted text and remove complete samples;
4. identify fuzzy text duplicates with MinHash and remove complete samples.

The original Nemotron Parse output remains immutable. PubMed uses the same flow but skips arXiv version selection.

## Artifact layout

```text
dedup/
  manifests/
  version_selection/
    baseline/
    inventory/
    removed_sample_ids/
    exceptions/
  sha_pdf/
    identification/
      inventory/
      duplicate_sample_ids/
    removal/deduplicated/
  exact_text/
    identification/
      ExactDuplicateIds/
      exact_id_generator.json
    removal/deduplicated/
  fuzzy/<config>/
    identification/
      cache/
      FuzzyDuplicateIds/
      fuzzy_id_generator.json
    removal/deduplicated/
```

Use a new fuzzy config directory whenever MinHash parameters change. Identification and removal must use the same
`--input-blocksize` or `--files-per-partition`; generated integer IDs are reconstructed from those exact file groups.

The interleaved data lives as flat `<hash>.parquet` files directly under the Nemotron Parse output directory. Both
`prepare.py` and `run.py` ignore any `.parquet`/`.jsonl` whose basename starts with `_` or `.` (for example the
`_perf_stats_<jobid>.parquet` and `_manifest_remaining.jsonl` sidecars the parse run writes alongside the data), so
the raw output directory can be passed as the input path without pre-cleaning.

## Prepare arXiv inventories

The metadata file must be a frozen, dated `arxiv-metadata-oai-snapshot.json` JSONL file (or an equivalent Parquet
snapshot) aligned as closely as possible to the downloaded PDF snapshot. Record its source and checksum alongside
the run artifacts.

fish (login):

```fish
set -x REPO /lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator
set -x PARSE /lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
set -x PDF_ROOT /lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv-pdfs
set -x DEDUP $PARSE/dedup
set -x ARXIV_METADATA /path/to/frozen/arxiv-metadata-oai-snapshot.json
set -x ARXIV_METADATA_SHA256 (sha256sum $ARXIV_METADATA | cut -d' ' -f1)
set -x ARXIV_METADATA_URL https://example.invalid/replace-with-actual-snapshot-url
set -x ARXIV_METADATA_DATE YYYY-MM-DD

cd $REPO
uv run --no-sync python tutorials/interleaved/deduplication/prepare.py baseline \
  --input-path $PARSE/out \
  --pdf-root $PDF_ROOT \
  --output-path $DEDUP/version_selection/baseline \
  --manifest-path $DEDUP/manifests/baseline.json

uv run --no-sync python tutorials/interleaved/deduplication/prepare.py arxiv-versions \
  --inventory-path $DEDUP/version_selection/baseline \
  --metadata-path $ARXIV_METADATA \
  --metadata-sha256 $ARXIV_METADATA_SHA256 \
  --metadata-source-url $ARXIV_METADATA_URL \
  --metadata-snapshot-date $ARXIV_METADATA_DATE \
  --output-path $DEDUP/version_selection/inventory \
  --removed-path $DEDUP/version_selection/removed_sample_ids \
  --exceptions-path $DEDUP/version_selection/exceptions \
  --manifest-path $DEDUP/manifests/version_selection.json
```

If `num_removed_samples` is zero, use `$PARSE/out` as the SHA input. Otherwise, first run `sample-id-remove` using
`version_selection/removed_sample_ids`.

## Run distributed stages

Ray jobs must use the activated environment directly; do not wrap these commands in `uv run`.

bash (Slurm node):

```bash
export REPO=/path/to/frozen/Curator
export CURATOR_ENV=/lustre/fsw/portfolios/nemotron/users/tromanski/github/Curator/.venv
export PARSE=/lustre/fsw/portfolios/nemotron/users/tromanski/workspace/arxiv_nemotron_parse_run
export PDF_ROOT=/lustre/fsw/portfolios/nemotron/users/tromanski/data/arxiv-pdfs
export DEDUP="$PARSE/dedup"
source "$CURATOR_ENV/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

python tutorials/interleaved/deduplication/run.py sha-inventory \
  --input-path "$PARSE/out" \
  --pdf-root "$PDF_ROOT" \
  --output-path "$DEDUP/sha_pdf/identification/inventory" \
  --manifest-path "$DEDUP/manifests/sha_pdf_inventory.json" \
  --input-blocksize 512MiB
```

Select SHA duplicate keepers after hashing:

fish (login):

```fish
uv run --no-sync python tutorials/interleaved/deduplication/prepare.py sha-select \
  --inventory-path $DEDUP/sha_pdf/identification/inventory \
  --output-path $DEDUP/sha_pdf/identification/duplicate_sample_ids \
  --manifest-path $DEDUP/manifests/sha_pdf.json
```

bash (Slurm node):

```bash
python tutorials/interleaved/deduplication/run.py sample-id-remove \
  --input-path "$PARSE/out" \
  --ids-to-remove-path "$DEDUP/sha_pdf/identification/duplicate_sample_ids" \
  --output-path "$DEDUP/sha_pdf/removal/deduplicated" \
  --manifest-path "$DEDUP/manifests/sha_pdf_removal.json" \
  --input-blocksize 512MiB

python tutorials/interleaved/deduplication/run.py exact-identify \
  --input-path "$DEDUP/sha_pdf/removal/deduplicated" \
  --output-path "$DEDUP/exact_text/identification" \
  --manifest-path "$DEDUP/manifests/exact_text_identification.json" \
  --input-blocksize 512MiB

python tutorials/interleaved/deduplication/run.py generated-id-remove \
  --input-path "$DEDUP/sha_pdf/removal/deduplicated" \
  --ids-to-remove-path "$DEDUP/exact_text/identification/ExactDuplicateIds" \
  --id-generator-path "$DEDUP/exact_text/identification/exact_id_generator.json" \
  --output-path "$DEDUP/exact_text/removal/deduplicated" \
  --manifest-path "$DEDUP/manifests/exact_text_removal.json" \
  --input-blocksize 512MiB

export FUZZY_CONFIG=ngram24_b20_h13
python tutorials/interleaved/deduplication/run.py fuzzy-identify \
  --input-path "$DEDUP/exact_text/removal/deduplicated" \
  --cache-path "$DEDUP/fuzzy/$FUZZY_CONFIG/identification/cache" \
  --output-path "$DEDUP/fuzzy/$FUZZY_CONFIG/identification" \
  --manifest-path "$DEDUP/manifests/fuzzy_$FUZZY_CONFIG.json" \
  --input-blocksize 512MiB \
  --char-ngrams 24 \
  --num-bands 20 \
  --minhashes-per-band 13

python tutorials/interleaved/deduplication/run.py generated-id-remove \
  --input-path "$DEDUP/exact_text/removal/deduplicated" \
  --ids-to-remove-path "$DEDUP/fuzzy/$FUZZY_CONFIG/identification/FuzzyDuplicateIds" \
  --id-generator-path "$DEDUP/fuzzy/$FUZZY_CONFIG/identification/fuzzy_id_generator.json" \
  --output-path "$DEDUP/fuzzy/$FUZZY_CONFIG/removal/deduplicated" \
  --manifest-path "${DEDUP}/manifests/fuzzy_${FUZZY_CONFIG}_removal.json" \
  --input-blocksize 512MiB
```

Before the full run, use a small frozen input directory to pilot all stages and inspect SHA, exact, and fuzzy duplicate
pairs. Run annotations and filters only from the final fuzzy-removal output.

After all passes complete, validate the sample-count identities:

fish (login):

```fish
uv run --no-sync python tutorials/interleaved/deduplication/prepare.py accounting \
  --baseline-manifest $DEDUP/manifests/baseline.json \
  --version-manifest $DEDUP/manifests/version_selection.json \
  --sha-manifest $DEDUP/manifests/sha_pdf.json \
  --exact-removal-manifest $DEDUP/manifests/exact_text_removal.json \
  --fuzzy-removal-manifest $DEDUP/manifests/fuzzy_$FUZZY_CONFIG"_removal.json" \
  --output-path $DEDUP/manifests/accounting.json
```
