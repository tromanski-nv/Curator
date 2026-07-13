# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tutorials.eai_crawl import run_slurm


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        s3_bucket="source",
        stream=True,
        checkpoint_path=str(tmp_path / "checkpoint"),
        s3_keys_file=str(tmp_path / "chunk.keys"),
        output_dir=str(tmp_path / "pdf"),
        cdx_output_dir=str(tmp_path / "cdx"),
        output_rclone_remote=None,
        slurm=False,
        backend="ray_actor_pool",
    )


def _patch_main(
    monkeypatch: pytest.MonkeyPatch,
    args: Namespace,
    events: list[str],
    *,
    pipeline_error: Exception | None = None,
) -> None:
    class _Client:
        ray_dashboard_port = None

        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            events.append("start")

        def stop(self) -> None:
            events.append("stop")

    pipeline = mock.Mock()
    pipeline.describe.return_value = "pipeline"
    if pipeline_error is None:
        pipeline.run.return_value = []
    else:
        pipeline.run.side_effect = pipeline_error

    monkeypatch.setattr(run_slurm, "parse_args", lambda: args)
    monkeypatch.setattr(run_slurm, "RayClient", _Client)
    monkeypatch.setattr(run_slurm, "manifest_identity", lambda _path: ("digest", 2))
    monkeypatch.setattr(run_slurm, "resolve_output_storage_options", lambda **_kwargs: {})
    monkeypatch.setattr(run_slurm, "initialize_output", lambda **_kwargs: True)
    monkeypatch.setattr(run_slurm, "build_pipeline", lambda _args: pipeline)
    monkeypatch.setattr(run_slurm, "build_executor", lambda _backend: object())
    monkeypatch.setattr(
        "nemo_curator.backends.failed_task_markers.failed_task_manifest_exists",
        lambda: False,
    )
    monkeypatch.setattr(
        "nemo_curator.backends.slurm_array.find_slurm_array_retries",
        lambda _path: SimpleNamespace(shard_indices=[]),
    )
    monkeypatch.setattr(
        run_slurm,
        "write_json",
        lambda _path, _marker, _options: events.append("marker"),
    )


def test_success_marker_is_written_only_after_ray_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    (tmp_path / "chunk.keys").write_text("one\ntwo\n")
    events: list[str] = []
    _patch_main(monkeypatch, args, events)

    assert run_slurm.main() == 0

    assert events == ["start", "stop", "marker"]


def test_pipeline_failure_stops_ray_without_success_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _args(tmp_path)
    (tmp_path / "chunk.keys").write_text("one\ntwo\n")
    events: list[str] = []
    _patch_main(monkeypatch, args, events, pipeline_error=RuntimeError("pipeline failed"))

    with pytest.raises(RuntimeError, match="pipeline failed"):
        run_slurm.main()

    assert events == ["start", "stop"]
