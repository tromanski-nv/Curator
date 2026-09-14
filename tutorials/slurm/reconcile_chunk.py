"""Read a chunk's output straight from S3 and reconcile sample_ids against its manifest."""
import sys, json, glob, os, subprocess, tempfile
import pyarrow.parquet as pq, pyarrow as pa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Single source of truth for the destination; do not restate it here.
from chunk_orchestrator import BUCKET, DEST_PREFIX, RCLONE_REMOTE as REMOTE
chunk=sys.argv[1]; W="/lustre/fsw/portfolios/nemotron/users/tromanski/scratch/nemotron_parse_production"
st=f"{W}/chunk_state/{chunk}"
fetched=json.load(open(f"{st}/_FETCHED")) if os.path.exists(f"{st}/_FETCHED") else {}
expect=fetched.get("fetched") or fetched.get("manifest_entries")
d=tempfile.mkdtemp(dir="/tmp")
rc=subprocess.run(["rclone","copy",
   f"{REMOTE}:{BUCKET}/{DEST_PREFIX}{chunk}/",d,
   "--transfers","32","--include","*.parquet"],capture_output=True,text=True)
fs=glob.glob(f"{d}/**/*.parquet",recursive=True)
ids=set(); rows=0
for f in fs:
    t=pq.read_table(f,columns=["sample_id"])
    rows+=t.num_rows; ids.update(t["sample_id"].to_pylist())
print(f"{chunk}: {len(fs)} parquet, {rows:,} rows, {len(ids):,} unique sample_id")
print(f"  manifest entries fetched: {expect}")
if expect:
    d_=expect-len(ids)
    print(f"  DELTA: {d_:+,}  ({'CLEAN' if d_==0 else 'MISSING '+str(d_)+' PDFs'})")
subprocess.run(["rm","-rf",d])

# NOTE: this exists because chunk_9000 passed every existing gate while silently
# losing a PDF. Its upload verified (object count + sizes matched), _PROCESSED and
# _UPLOADED were written, and it was reaped -- yet its output holds 4,982 unique
# sample_ids against 4,983 fetched. Nothing in the pipeline compares those two
# numbers, so the loss was invisible. Across the corpus that rate is many thousands of
# documents, discovered only long after the inputs are gone.
#
# Run before reaping a chunk:
#   python tutorials/slurm/reconcile_chunk.py chunk_0000
# A non-zero DELTA means output is incomplete: do NOT reap, investigate first.
