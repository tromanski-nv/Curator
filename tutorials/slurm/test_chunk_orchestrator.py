#!/usr/bin/env python3
"""Regression tests for chunk_orchestrator.py's safety guards.

These cover the properties that, if they broke, would either destroy the
source corpus or wedge the login node. They touch no network, no
Slurm and no real data, so they are safe to run anywhere::

    $PY -m unittest discover -s tutorials/slurm -p 'test_chunk_orchestrator.py' -v
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chunk_orchestrator as co  # noqa: E402

GOOD = "pdf_corpus/v1/nemotron_parse_elements/chunk_0000/"


class TestRemoteGuard(unittest.TestCase):
    def test_accepts_own_prefix(self):
        co.assert_safe_remote(co.BUCKET, GOOD, 0)
        co.assert_safe_remote(co.BUCKET, co.DEST_PREFIX, None)

    def test_rejects_source_corpus(self):
        for bad in ("pdf_corpus/v1/data/", "pdf_corpus/v1/index/",
                    "pdf_corpus/v1/errors/", "pdf_corpus/v1/markers/",
                    "pdf_corpus/v1/", "", "/"):
            with self.subTest(prefix=bad), self.assertRaises(SystemExit):
                co.assert_safe_remote(co.BUCKET, bad, None)

    def test_rejects_wrong_bucket(self):
        with self.assertRaises(SystemExit):
            co.assert_safe_remote("some-other-bucket", GOOD, 0)

    def test_rejects_traversal(self):
        for bad in (co.DEST_PREFIX + "../data/", co.DEST_PREFIX + "a//b/"):
            with self.subTest(prefix=bad), self.assertRaises(SystemExit):
                co.assert_safe_remote(co.BUCKET, bad, None)

    def test_rejects_other_chunk(self):
        """A chunk may only ever be told to write its own prefix."""
        with self.assertRaises(SystemExit):
            co.assert_safe_remote(co.BUCKET, GOOD, 1)

    def test_token_alone_is_not_enough(self):
        with self.assertRaises(SystemExit):
            co.assert_safe_remote(co.BUCKET, "pdf_corpus/v2/nemotron_parse_elements/", None)


class TestVerbGuard(unittest.TestCase):
    def test_allows_copy_semantics(self):
        co.assert_no_destructive_verb(["rclone", "check", "a", "b"])
        co.assert_no_destructive_verb([co.DM_BIN, "job", "copy", "a", "b"])

    def test_rejects_destructive(self):
        for verb in ("sync", "purge", "delete", "deletefile", "rmdir", "rmdirs",
                     "cleanup", "move", "moveto"):
            with self.subTest(verb=verb), self.assertRaises(SystemExit):
                co.assert_no_destructive_verb(["rclone", verb, "a", "b"])


class TestLocalDeleteGuard(unittest.TestCase):
    def test_allows_own_chunk_dirs(self):
        self.assertTrue(co.assert_safe_local_delete(co.stage_dir(7), 7))
        self.assertTrue(co.assert_safe_local_delete(co.out_dir(7), 7))

    def test_rejects_roots_and_neighbours(self):
        for path, chunk in ((co.STAGE_ROOT, 7), (co.OUT_ROOT, 7), (co.WORK, 7),
                            (co.STATE_ROOT, 7), (co.BASE, 7), ("/", 7),
                            (os.path.expanduser("~"), 7), (co.INDEX_DIR, 7),
                            (co.state_dir(7), 7), (co.stage_dir(8), 7)):
            with self.subTest(path=path), self.assertRaises(SystemExit):
                co.assert_safe_local_delete(path, chunk)


class TestChunkNaming(unittest.TestCase):
    def test_padding(self):
        self.assertEqual(co.cid(0), "chunk_0000")
        self.assertEqual(co.cid(9999), "chunk_9999")

    def test_rejects_out_of_range(self):
        for bad in (-1, 10000, True, 1.0, "0"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                co.cid(bad)


class TestPartition(unittest.TestCase):
    rows = [{"tar_key": f"t{i:03d}", "index_file": f"f{i}", "n_pdfs": 100, "bytes": 1000}
            for i in range(7)]

    def test_tars_per_chunk(self):
        chunks = co.partition(self.rows, None, 2, None, 0)
        self.assertEqual([c["chunk_id"] for c in chunks], [0, 1, 2, 3])
        self.assertEqual([len(c["tars"]) for c in chunks], [2, 2, 2, 1])

    def test_deterministic(self):
        self.assertEqual(co.partition(self.rows, None, 2, None, 0),
                         co.partition(self.rows, None, 2, None, 0))

    def test_chunk_id_base_offsets_ids(self):
        chunks = co.partition(self.rows, None, 2, None, 9000)
        self.assertEqual(chunks[0]["chunk_id"], 9000)

    def test_target_bytes(self):
        chunks = co.partition(self.rows, 2500, None, None, 0)
        self.assertEqual([len(c["tars"]) for c in chunks], [3, 3, 1])

    def test_max_pdfs_marks_partial(self):
        chunks = co.partition(self.rows, None, 3, 250, 0)
        self.assertTrue(chunks[0]["partial_coverage"])
        self.assertEqual(chunks[0]["n_pdfs"], 250)


class TestSizing(unittest.TestCase):
    def test_pdfs_per_shard_is_sized_from_observed_shards_not_a_formula(self):
        """The formula 6.59*4*9500/14.53 = 17,235 was what chunk_0000 ran, and it
        was too large: 2 of its 8 shards hit the 9,900s timeout. The formula
        assumes the mean rate applies to every shard, but a chunk is 64
        CONTIGUOUS tars, so page counts are correlated and the observed max/min
        spread was 1.41x -- not the 1.069 an independent-sampling model predicts.
        11,500 is sized so the SLOWEST shard actually measured (19.6 pages/s/node)
        finishes inside the budget with ~13% headroom."""
        self.assertEqual(co.PDFS_PER_SHARD, 11500)
        worst_rate_pages_per_s = 19.6          # chunk_0000 shard 5
        startup_s = 130
        projected = co.PDFS_PER_SHARD * 14.53 / worst_rate_pages_per_s + startup_s
        self.assertLess(projected, 9900, "worst observed shard must fit the budget")
        self.assertGreater(9900 - projected, 0.10 * 9900, "want >10% headroom")

    def test_shard_count_is_a_ceiling_division(self):
        """A partial final shard still gets a shard; an exact multiple does not
        get a spare. Stated as a property so it does not encode a corpus size."""
        n = co.PDFS_PER_SHARD
        self.assertEqual(-(-(n * 4) // n), 4, "exact multiple must not add a shard")
        self.assertEqual(-(-(n * 4 + 1) // n), 5, "one leftover doc needs one more shard")
        self.assertEqual(-(-1 // n), 1, "a single document still needs a shard")


class TestLocalBudget(unittest.TestCase):
    def test_clamp_off_allocation(self):
        saved = os.environ.pop("SLURM_JOB_ID", None)
        try:
            self.assertEqual(co.clamp_workers(64, "t"), co.MAX_LOCAL_WORKERS)
            self.assertEqual(co.clamp_workers(2, "t"), 2)
            self.assertEqual(co.clamp_workers(0, "t"), 1)
        finally:
            if saved is not None:
                os.environ["SLURM_JOB_ID"] = saved

    def test_clamp_in_allocation(self):
        os.environ["SLURM_JOB_ID"] = "1"
        try:
            self.assertEqual(co.clamp_workers(256, "t"), co.MAX_ALLOC_WORKERS)
        finally:
            del os.environ["SLURM_JOB_ID"]

    def test_headroom_refuses_an_impossible_pool(self):
        with self.assertRaises(SystemExit):
            co.assert_nproc_headroom(100_000, "deliberately absurd pool")

    def test_user_thread_count_is_plausible(self):
        n = co.user_thread_count()
        self.assertGreater(n, 0)
        self.assertLess(n, 100_000)

    def test_own_tree_is_a_subset_of_the_user_total(self):
        """The hard guard is scoped to our own subtree; an earlier version
        aborted on the user total and killed a run whose own tree held ~3
        threads, seconds before it would have written _PROCESSED."""
        tree, nproc = co.own_tree_threads()
        total = co.user_thread_count()
        self.assertGreaterEqual(tree, 1)
        self.assertGreaterEqual(nproc, 1)
        self.assertLessEqual(tree, total)

    def test_own_tree_counts_descendants(self):
        kid = co.popen_child(["sleep", "30"])
        try:
            time.sleep(0.5)
            tree, nproc = co.own_tree_threads()
            self.assertGreaterEqual(nproc, 2, "child not counted in the tree")
        finally:
            co.kill_children()
            co.forget_child(kid)


class TestChildSupervision(unittest.TestCase):
    def test_child_dies_with_a_sigkilled_parent(self):
        """PR_SET_PDEATHSIG: the orphan-after-28-minutes regression."""
        script = (
            "import os,subprocess,sys,time\n"
            f"sys.path.insert(0,{os.path.dirname(os.path.abspath(__file__))!r})\n"
            "import chunk_orchestrator as co\n"
            "p=co.popen_child(['sleep','600'])\n"
            "print(p.pid,flush=True)\n"
            "time.sleep(600)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            grandchild = int(parent.stdout.readline().strip())
            os.kill(parent.pid, 9)  # SIGKILL: no handler can run
            parent.wait(timeout=10)
            for _ in range(100):
                if not os.path.exists(f"/proc/{grandchild}"):
                    break
                time.sleep(0.1)
            self.assertFalse(os.path.exists(f"/proc/{grandchild}"),
                             f"pid {grandchild} outlived its SIGKILLed parent")
        finally:
            if parent.poll() is None:
                parent.kill()

    def test_child_dies_on_sigterm_to_the_parent(self):
        script = (
            "import os,signal,subprocess,sys,time\n"
            f"sys.path.insert(0,{os.path.dirname(os.path.abspath(__file__))!r})\n"
            "import chunk_orchestrator as co\n"
            "co.install_signal_handlers()\n"
            "p=co.popen_child(['sleep','600'])\n"
            "print(p.pid,flush=True)\n"
            "time.sleep(600)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            grandchild = int(parent.stdout.readline().strip())
            os.kill(parent.pid, 15)
            parent.wait(timeout=20)
            for _ in range(100):
                if not os.path.exists(f"/proc/{grandchild}"):
                    break
                time.sleep(0.1)
            self.assertFalse(os.path.exists(f"/proc/{grandchild}"))
        finally:
            if parent.poll() is None:
                parent.kill()

    def test_run_cmd_returns_output(self):
        proc = co.run_cmd(["echo", "hello"], check=True)
        self.assertEqual(proc.stdout.strip(), "hello")

    def test_run_cmd_raises_on_failure(self):
        with self.assertRaises(SystemExit):
            co.run_cmd(["false"])


class TestDirMarkers(unittest.TestCase):
    """DataMover writes a zero-byte object per directory. Measured against the
    live bucket: a 2-file copy produced 4 keys, two of them markers."""

    LISTING = {
        "p/chunk_9999/": {"size": 0, "etag": "d41d8cd98f00b204e9800998ecf8427e"},
        "p/chunk_9999/shard_0000/": {"size": 0, "etag": "d41d8cd98f00b204e9800998ecf8427e"},
        "p/chunk_9999/shard_0000/a.parquet": {"size": 1048576, "etag": "b7f1"},
        "p/chunk_9999/shard_0000/b.parquet": {"size": 524288, "etag": "51ed"},
    }

    def test_markers_are_separated(self):
        real, markers = co.strip_dir_markers(self.LISTING)
        self.assertEqual(len(real), 2)
        self.assertEqual(markers, ["p/chunk_9999/", "p/chunk_9999/shard_0000/"])

    def test_zero_byte_file_is_not_a_marker(self):
        """An empty output file must still be verified, not silently dropped."""
        real, markers = co.strip_dir_markers({"p/c/empty.parquet": {"size": 0, "etag": "x"}})
        self.assertEqual(list(real), ["p/c/empty.parquet"])
        self.assertEqual(markers, [])

    def test_nonempty_key_ending_in_slash_is_kept(self):
        real, markers = co.strip_dir_markers({"p/c/odd/": {"size": 5, "etag": "x"}})
        self.assertEqual(list(real), ["p/c/odd/"])
        self.assertEqual(markers, [])


class TestArrayFormatting(unittest.TestCase):
    def test_ranges(self):
        self.assertEqual(co.format_array_indices([0, 1, 2, 5, 7, 8]), "0-2,5,7-8")
        self.assertEqual(co.format_array_indices([]), "")
        self.assertEqual(co.format_array_indices([3]), "3")


class TestSqueueFailureIsNotCompletion(unittest.TestCase):
    """squeue's exit code cannot tell "job finished" from "slurm is broken".

    `squeue -j <id>` exits 1 with "Invalid job id specified" both for a job that
    has left the queue and for an id that never existed (verified on this
    cluster against finished array 721182), so ONLY the error text separates
    those from a transport failure. Reading a transport failure as "drained"
    makes cmd_process rebuild the retry plan from partial on-disk manifests and
    resubmit shards that are still running -- two arrays writing the same paths
    on 8 GPU nodes each. The drain loop polls very many times across the campaign, so this is a when, not an if.
    """

    def setUp(self):
        self._run, self._sleep, self._log = co.run_cmd, time.sleep, co.log
        time.sleep = lambda *_a, **_k: None
        co.log = lambda *_a, **_k: None

    def tearDown(self):
        co.run_cmd, time.sleep, co.log = self._run, self._sleep, self._log

    def _fake(self, rc, stdout="", stderr=""):
        calls = []

        def run_cmd(argv, **_kw):
            calls.append(argv)
            return subprocess.CompletedProcess(argv, rc, stdout, stderr)

        co.run_cmd = run_cmd
        return calls

    def test_finished_job_reads_as_absent_not_unknown(self):
        """The normal path. If this regresses, every chunk hangs to its timeout."""
        self._fake(1, stderr="slurm_load_jobs error: Invalid job id specified")
        self.assertEqual(co.squeue_alive("721182"), [])

    def test_transport_failure_is_unknown_not_absent(self):
        """The bug. [] here resubmits a live 8-node array."""
        for err in ("slurm_load_jobs error: Socket timed out on send/recv operation",
                    "slurm_load_jobs error: Unable to contact slurm controller (connect failure)",
                    "slurm_load_jobs error: Zero Bytes were transmitted or received"):
            with self.subTest(err=err):
                self._fake(1, stderr=err)
                self.assertIsNone(co.squeue_alive("721182"), f"{err!r} must not read as finished")

    def test_transport_failure_is_retried_before_giving_up(self):
        calls = self._fake(1, stderr="Socket timed out on send/recv operation")
        self.assertIsNone(co.squeue_alive("721182"))
        self.assertEqual(len(calls), co.SQUEUE_TRIES)

    def test_absent_marker_short_circuits_retries(self):
        """A finished job must not cost SQUEUE_TRIES x SQUEUE_RETRY_S on every poll."""
        calls = self._fake(1, stderr="slurm_load_jobs error: Invalid job id specified")
        self.assertEqual(co.squeue_alive("721182"), [])
        self.assertEqual(len(calls), 1)

    def test_live_job_returns_rows(self):
        self._fake(0, stdout="721182_0 RUNNING\n721182_1 PENDING\n")
        self.assertEqual(len(co.squeue_alive("721182")), 2)

    def test_by_name_transport_failure_is_unknown(self):
        """None here, not [] -- [] means "no orphans" and permits a double-submit."""
        self._fake(1, stderr="slurm_load_jobs error: Socket timed out on send/recv operation")
        self.assertIsNone(co.squeue_by_name("parse-chunk_0001"))

    def test_sacct_unknown_job_is_unknown_not_terminal(self):
        self._fake(0, stdout="")
        self.assertIsNone(co.sacct_live_tasks("999999999"))

    def test_sacct_separates_live_from_terminal(self):
        self._fake(0, stdout="721182_0|COMPLETED\n721182_1|RUNNING\n"
                             "721182_2|CANCELLED by 12345\n721182_3|PENDING\n")
        self.assertEqual(co.sacct_live_tasks("721182"), ["721182_1", "721182_3"])

    def test_sacct_all_terminal_is_empty(self):
        self._fake(0, stdout="721182_0|COMPLETED\n721182_1|FAILED\n"
                             "721182_2|TIMEOUT\n721182_3|NODE_FAIL\n")
        self.assertEqual(co.sacct_live_tasks("721182"), [])


class TestPipelinedRun(unittest.TestCase):
    """_run_stages must overlap chunks and survive a failing one.

    Sequentially the campaign is the sum of every chunk, so the campaign only
    finishes if chunks overlap; and it only finishes UNATTENDED if one chunk's
    SystemExit does not abandon the other 1,048.
    """

    def setUp(self):
        self._fn = dict(co.STAGE_FN)
        self._log, self._has, self._read = co.log, co.has_marker, co.read_marker
        co.log = lambda *_a, **_k: None
        co.has_marker = lambda *_a, **_k: False
        # read_marker must be stubbed too, not just has_marker: these chunk ids
        # exist on the real cluster, and their _PLANNED records a plan_dir that
        # would trip the ownership guard before any stage ran. A unit test must
        # not depend on what is currently staged on Lustre.
        co.read_marker = lambda *_a, **_k: {}

    def tearDown(self):
        co.STAGE_FN.clear()
        co.STAGE_FN.update(self._fn)
        co.log, co.has_marker, co.read_marker = self._log, self._has, self._read

    def _args(self, k, ids):
        return argparse.Namespace(
            plan_dir="/nonexistent-plan-dir", dry_run=True, max_in_flight=k,
            stages="fetch", chunks=",".join(str(i) for i in ids))

    def _plan(self, ids):
        return {i: {"tars": [], "n_pdfs": 1, "input_bytes": 1, "chunk_id": i} for i in ids}

    def test_failure_of_one_chunk_does_not_stop_the_rest(self):
        seen = []
        lock = threading.Lock()

        def fetch(sub):
            with lock:
                seen.append(sub.chunk)
            if sub.chunk == 3:  # noqa: PLR2004
                msg = "synthetic refusal"
                raise SystemExit(msg)

        co.STAGE_FN["fetch"] = fetch
        ids = [1, 2, 3, 4, 5]
        co._run_stages(self._args(2, ids), self._plan(ids), ids, ["fetch"])
        self.assertEqual(sorted(seen), ids, "every chunk must be attempted")

    def test_never_exceeds_k_concurrent_chunks(self):
        """K bounds Lustre footprint: each in-flight chunk holds ~405 GiB."""
        live = 0
        peak = 0
        lock = threading.Lock()

        def fetch(_sub):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            with lock:
                live -= 1

        co.STAGE_FN["fetch"] = fetch
        ids = list(range(1, 13))
        co._run_stages(self._args(3, ids), self._plan(ids), ids, ["fetch"])
        self.assertLessEqual(peak, 3, f"peak {peak} concurrent chunks exceeds K=3")
        self.assertGreater(peak, 1, "chunks did not overlap at all -- still sequential?")

    def test_k_of_one_is_sequential(self):
        live = 0
        peak = 0

        def fetch(_sub):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            live -= 1

        co.STAGE_FN["fetch"] = fetch
        ids = [1, 2, 3]
        co._run_stages(self._args(1, ids), self._plan(ids), ids, ["fetch"])
        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
