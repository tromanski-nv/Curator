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

"""Tests for Nemotron-Parse inference-server configuration."""

import pytest

from nemo_curator.core.serve import DynamoServerConfig, DynamoVLLMModelConfig, RayServeModelConfig
from nemo_curator.core.serve.dynamo.vllm import dynamo_runtime_env, merge_model_runtime_envs
from nemo_curator.stages.interleaved.pdf.nemotron_parse.server import create_nemotron_parse_inference_server


def test_dynamo_server_has_pdf_defaults_and_overrides() -> None:
    server = create_nemotron_parse_inference_server(
        model_path="/models/nemotron-parse",
        model_name="nemotron-parse",
        num_replicas=2,
        engine_kwargs={"enforce_eager": True},
        request_timeout_s=123,
    )

    model = server.models[0]
    assert isinstance(model, DynamoVLLMModelConfig)
    assert model.model_identifier == "/models/nemotron-parse"
    assert model.model_name == "nemotron-parse"
    assert model.num_replicas == 2
    assert model.engine_kwargs["limit_mm_per_prompt"] == {"image": 1}
    assert model.engine_kwargs["enforce_eager"] is True
    assert model.dynamo_kwargs == {"enable_multimodal": True}
    assert model.runtime_env == {"uv": {"packages": ["albumentations==2.0.8"]}}
    assert isinstance(server.backend, DynamoServerConfig)
    assert server.backend.request_plane == "tcp"
    assert server.backend.router.router_kwargs == {"trust_remote_code": True}
    assert server.backend.subprocess_env == {"DYN_TCP_REQUEST_TIMEOUT": "123"}


def test_ray_serve_server_has_pdf_defaults() -> None:
    server = create_nemotron_parse_inference_server(backend="ray-serve", num_replicas=3)

    model = server.models[0]
    assert isinstance(model, RayServeModelConfig)
    assert model.deployment_config == {"num_replicas": 3}
    assert model.engine_kwargs["limit_mm_per_prompt"] == {"image": 1}


def test_runtime_env_defaults_to_the_actor_venv_install() -> None:
    """No caller runtime_env -> Ray still builds and populates the actor venv."""
    server = create_nemotron_parse_inference_server(num_replicas=1)

    assert server.models[0].runtime_env == {"uv": {"packages": ["albumentations==2.0.8"]}}
    merged = dynamo_runtime_env(server.models[0])
    assert "py_executable" not in merged
    assert merged["uv"]["packages"][-1] == "albumentations==2.0.8"
    assert any(p.startswith("ai-dynamo") for p in merged["uv"]["packages"])


def test_py_executable_suppresses_the_actor_venv_install() -> None:
    """A prebaked venv must not also pay for the ~247-package uv resolve."""
    server = create_nemotron_parse_inference_server(
        num_replicas=2,
        runtime_env={"py_executable": "/opt/dynamo-pdf/bin/python"},
    )

    model = server.models[0]
    assert model.runtime_env == {"py_executable": "/opt/dynamo-pdf/bin/python"}

    # Both the per-worker and the shared-frontend merges must drop the
    # DYNAMO_VLLM_RUNTIME_ENV uv block, not just the stage-level one.
    for merged in (dynamo_runtime_env(model), merge_model_runtime_envs([model])):
        assert merged["py_executable"] == "/opt/dynamo-pdf/bin/python"
        assert "uv" not in merged
        assert "pip" not in merged
        assert merged["config"] == {"setup_timeout_seconds": 600}


def test_runtime_env_env_vars_reach_the_workers() -> None:
    server = create_nemotron_parse_inference_server(
        num_replicas=1,
        runtime_env={"env_vars": {"CUDA_CACHE_PATH": "/lustre/cuda_cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"}},
    )

    merged = dynamo_runtime_env(server.models[0])
    assert merged["env_vars"] == {"CUDA_CACHE_PATH": "/lustre/cuda_cache", "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    # env_vars alone must NOT suppress the install.
    assert "uv" in merged


def test_ray_serve_backend_honours_py_executable() -> None:
    server = create_nemotron_parse_inference_server(
        backend="ray-serve",
        num_replicas=1,
        runtime_env={"py_executable": "/opt/dynamo-pdf/bin/python"},
    )

    assert server.models[0].runtime_env == {"py_executable": "/opt/dynamo-pdf/bin/python"}


@pytest.mark.parametrize("num_replicas", [0, -1])
def test_rejects_non_positive_replica_count(num_replicas: int) -> None:
    with pytest.raises(ValueError, match="num_replicas must be at least 1"):
        create_nemotron_parse_inference_server(num_replicas=num_replicas)
