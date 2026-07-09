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

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tutorials/eai_crawl/run_day_array.sh"
BASH = "/bin/bash"


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _launcher_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "rclone",
        """#!/bin/bash
set -eu
printf '%s\n' "$*" >> "$RCLONE_CALL_LOG"
if [[ "$*" == *"team-vendor-data:"* ]]; then
    if [[ "${RCLONE_FAIL_SOURCE:-0}" == "1" ]]; then
        exit 9
    fi
    printf '%s\n' \
        z.warc.gz \
        b.warc.gz \
        readme.txt \
        g.warc.gz \
        a.warc.gz \
        f.warc.gz \
        c.warc.gz \
        d.warc.gz
fi
""",
    )
    _write_executable(
        bin_dir / "scontrol",
        """#!/bin/bash
echo 'MaxArraySize            = 1001'
""",
    )
    _write_executable(
        bin_dir / "srun",
        """#!/bin/bash
set -eu
printf '%s\n' "$*" > "$SRUN_CALL_LOG"
""",
    )
    _write_executable(
        bin_dir / "sbatch",
        """#!/bin/bash
set -eu
printf '%s\n' "$*" > "$SBATCH_CALL_LOG"
echo 'Submitted batch job 42'
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "AWS_ACCESS_KEY_ID": "test-read-key",
            "AWS_SECRET_ACCESS_KEY": "test-read-secret",
            "CURATOR_DIR": str(REPO_ROOT),
            "WORKLIST_ROOT": str(tmp_path / "worklists"),
            "LOG_ROOT": str(tmp_path / "logs"),
            "RCLONE_CALL_LOG": str(tmp_path / "rclone_calls.txt"),
            "SBATCH_CALL_LOG": str(tmp_path / "sbatch_call.txt"),
            "WARCS_PER_ARRAY_TASK": "3",
            "DRY_RUN": "1",
        }
    )
    return env


def _submit_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    env = _launcher_env(tmp_path)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/activate").touch()
    key_dir = tmp_path / "worklists/20240814.test"
    key_dir.mkdir(parents=True)
    (key_dir / "group_00007.txt").write_text("eai-warc/20240814/a.warc.gz\n", encoding="utf-8")
    env.update(
        {
            "VENV_PATH": str(venv),
            "SRUN_CALL_LOG": str(tmp_path / "srun_call.txt"),
            "SLURM_JOB_ID": "456",
            "SLURM_ARRAY_JOB_ID": "123",
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_JOB_NODELIST": "cpu-node-01",
            "SLURM_CPUS_PER_TASK": "64",
            "EAI_WARC_KEY_DIR": str(key_dir),
            "EAI_S3_BUCKET": "source-bucket",
            "EAI_S3_PREFIX": "eai-warc/20240814/",
            "EAI_S3_ENDPOINT_URL": "https://object-store.example",
            "EAI_STREAM": "1",
            "EAI_OUTPUT_DIR": "s3://output/pdf/crawl_date=20240814/",
            "EAI_CDX_OUTPUT_DIR": "s3://output/cdx/crawl_date=20240814/",
            "EAI_OUTPUT_RCLONE_REMOTE": "output-remote",
        }
    )
    return env, key_dir


def test_groups_one_day_once_and_builds_one_node_array(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path)

    result = subprocess.run(  # noqa: S603
        [BASH, str(LAUNCHER), "20240814"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    worklist_dir = tmp_path / "worklists/20240814.active"
    group_files = sorted(worklist_dir.glob("group_*.txt"))
    assert [path.read_text(encoding="utf-8").splitlines() for path in group_files] == [
        ["eai-warc/20240814/a.warc.gz", "eai-warc/20240814/b.warc.gz", "eai-warc/20240814/c.warc.gz"],
        ["eai-warc/20240814/d.warc.gz", "eai-warc/20240814/f.warc.gz", "eai-warc/20240814/g.warc.gz"],
        ["eai-warc/20240814/z.warc.gz"],
    ]
    assert all(path.stat().st_mode & 0o222 == 0 for path in group_files)

    calls = (tmp_path / "rclone_calls.txt").read_text(encoding="utf-8").splitlines()
    source_calls = [call for call in calls if "team-vendor-data:" in call]
    assert len(source_calls) == 1
    assert "WARCs     : 7" in result.stdout
    assert "Groups    : 3 (<= 3 WARCs each)" in result.stdout
    assert "--nodes=1" in result.stdout
    assert "--exclusive" in result.stdout
    assert "--array=0-2%108" in result.stdout
    assert "--no-requeue" in result.stdout

    env["DRY_RUN"] = "0"
    env["EXISTING_WORKLIST_DIR"] = str(worklist_dir)
    submitted = subprocess.run(  # noqa: S603
        [BASH, str(LAUNCHER), "20240814"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    calls = (tmp_path / "rclone_calls.txt").read_text(encoding="utf-8").splitlines()
    source_calls = [call for call in calls if "team-vendor-data:" in call]
    assert len(source_calls) == 1
    assert "Submitted batch job 42" in submitted.stdout
    assert (worklist_dir / ".submitted/sbatch.out").read_text(encoding="utf-8") == "Submitted batch job 42\n"

    duplicate = subprocess.run(  # noqa: S603
        [BASH, str(LAUNCHER), "20240814"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert duplicate.returncode != 0
    assert "already been submitted" in duplicate.stderr


def test_source_listing_failure_does_not_leave_a_worklist(tmp_path: Path) -> None:
    env = _launcher_env(tmp_path)
    env["RCLONE_FAIL_SOURCE"] = "1"

    result = subprocess.run(  # noqa: S603
        [BASH, str(LAUNCHER), "20240814"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "failed to list and group WARCs" in result.stderr
    assert not (tmp_path / "worklists/20240814.active").exists()


def test_submit_resolves_one_group_and_isolates_outputs(tmp_path: Path) -> None:
    env, key_dir = _submit_env(tmp_path)

    subprocess.run(  # noqa: S603
        [BASH, str(REPO_ROOT / "tutorials/eai_crawl/submit.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    srun_call = (tmp_path / "srun_call.txt").read_text(encoding="utf-8")
    assert f"--s3-key-file '{key_dir}/group_00007.txt'" in srun_call
    assert "--output-dir 's3://output/pdf/crawl_date=20240814/warc_group=00007/'" in srun_call
    assert "--cdx-output-dir s3://output/cdx/crawl_date=20240814/warc_group=00007/" in srun_call
    assert "export RAY_TMPDIR='/tmp/ray_123_7'" in srun_call
    assert "--ray-num-cpus 64" in srun_call
    assert "--slurm" not in srun_call


def test_submit_rejects_array_worklist_with_local_source(tmp_path: Path) -> None:
    env, _ = _submit_env(tmp_path)
    env.pop("EAI_S3_BUCKET")
    env["EAI_WARC_DIR"] = "/local/warcs"

    result = subprocess.run(  # noqa: S603
        [BASH, str(REPO_ROOT / "tutorials/eai_crawl/submit.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "array worklists require S3 streaming" in result.stderr
    assert not (tmp_path / "srun_call.txt").exists()
