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

"""Inference-server configuration for Nemotron-Parse."""

from __future__ import annotations

from typing import Any, Literal

from nemo_curator.core.serve import (
    DynamoRouterConfig,
    DynamoServerConfig,
    DynamoVLLMModelConfig,
    InferenceServer,
    RayServeModelConfig,
)
from nemo_curator.core.serve.base import BaseModelConfig
from nemo_curator.stages.interleaved.pdf.nemotron_parse.inference import DEFAULT_MODEL_PATH

NemotronParseServerBackend = Literal["ray-serve", "dynamo"]

_DEFAULT_ENGINE_KWARGS: dict[str, Any] = {
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "limit_mm_per_prompt": {"image": 1},
    "enable_prefix_caching": False,
    "disable_hybrid_kv_cache_manager": False,
}

# Installed into the Ray actor venv when it is built on the fly. A prebaked
# ``py_executable`` venv is expected to already carry it -- see
# :func:`_resolve_runtime_env`.
_DEFAULT_RUNTIME_ENV: dict[str, Any] = {"uv": {"packages": ["albumentations==2.0.8"]}}


def _resolve_runtime_env(runtime_env: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a caller ``runtime_env`` over the stage default.

    A caller-supplied ``py_executable`` points Ray at an interpreter that
    already exists (a venv prebaked into the container image), so Ray skips
    building an actor venv entirely. Ray's ``uv``/``pip`` plugins, however,
    still run and still create+populate one -- and then ``py_executable``
    overrides ``context.py_executable`` afterwards, so the minutes spent
    resolving those packages are thrown away, and the interpreter that
    actually runs never sees them. Drop the package spec instead of paying
    for it: the prebaked venv is responsible for its own contents.

    This deliberately also removes the ``uv`` block contributed by
    :data:`nemo_curator.core.serve.dynamo.vllm.DYNAMO_VLLM_RUNTIME_ENV`;
    that merge happens later, in ``dynamo_runtime_env()``, which applies the
    same rule.
    """
    merged = BaseModelConfig.merge_runtime_envs(_DEFAULT_RUNTIME_ENV, runtime_env or None)
    if merged.get("py_executable"):
        merged.pop("uv", None)
        merged.pop("pip", None)
    return merged


def create_nemotron_parse_inference_server(  # noqa: PLR0913
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    model_name: str | None = None,
    backend: NemotronParseServerBackend = "dynamo",
    num_replicas: int = 1,
    engine_kwargs: dict[str, Any] | None = None,
    request_timeout_s: float = 300.0,
    health_check_timeout_s: int = 900,
    runtime_env: dict[str, Any] | None = None,
) -> InferenceServer:
    """Return an inference server configured for Nemotron-Parse PDFs.

    The returned server is not started. Use it as a context manager or call
    :meth:`InferenceServer.start` and :meth:`InferenceServer.stop` explicitly.

    Args:
        runtime_env: Optional Ray ``runtime_env`` merged over the stage
            default. Two keys matter here:

            ``py_executable``
                Path to an interpreter Ray should launch the model workers
                with, e.g. ``{"py_executable": "/opt/dynamo-pdf/bin/python"}``
                for an image that prebakes the Dynamo/vLLM venv. Supplying it
                suppresses the ``uv`` package install (see
                :func:`_resolve_runtime_env`), turning a multi-minute
                per-job dependency resolution into a no-op.
            ``env_vars``
                Scoped to the model's workers rather than the whole Slurm
                job -- the right home for e.g. ``CUDA_CACHE_PATH`` and
                ``VLLM_USE_FLASHINFER_SAMPLER``.

            Omit it for the previous behaviour: Ray clones the driver venv
            and installs Dynamo's vLLM extra into it on every job.
    """
    if num_replicas < 1:
        msg = f"num_replicas must be at least 1, got {num_replicas}"
        raise ValueError(msg)
    if request_timeout_s < 1:
        msg = f"request_timeout_s must be at least 1, got {request_timeout_s}"
        raise ValueError(msg)

    resolved_engine_kwargs = {**_DEFAULT_ENGINE_KWARGS, **(engine_kwargs or {})}
    model_kwargs = {
        "model_identifier": model_path,
        "model_name": model_name,
        "engine_kwargs": resolved_engine_kwargs,
        "runtime_env": _resolve_runtime_env(runtime_env),
    }

    if backend == "dynamo":
        model = DynamoVLLMModelConfig(
            **model_kwargs,
            num_replicas=num_replicas,
            dynamo_kwargs={"enable_multimodal": True},
        )
        server_config = DynamoServerConfig(
            request_plane="tcp",
            router=DynamoRouterConfig(router_kwargs={"trust_remote_code": True}),
            subprocess_env={"DYN_TCP_REQUEST_TIMEOUT": str(int(request_timeout_s))},
        )
        return InferenceServer(
            models=[model],
            backend=server_config,
            health_check_timeout_s=health_check_timeout_s,
        )
    if backend == "ray-serve":
        model = RayServeModelConfig(
            **model_kwargs,
            deployment_config={"num_replicas": num_replicas},
        )
        return InferenceServer(models=[model], health_check_timeout_s=health_check_timeout_s)

    msg = f"Unsupported inference server backend: {backend}"
    raise ValueError(msg)
