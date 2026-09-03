"""TensorRT backend for pi05 ``sample_actions``.

``PI0TensorRT`` stands in for ``PI0Pytorch`` inside ``openpi.policies.policy.Policy``:
same ``sample_actions`` signature (so ``serve_policy.py``'s RTC detection and
warmup work unchanged), same ``action_horizon`` / ``action_dim`` attributes and
the same observation preprocessing. The engine is a single graph over flat
tensors covering prefix encoding and all denoising steps with trained-RTC
inputs; ``inference_delay = 0`` is the plain (non-RTC) request. This module owns
the observation -> engine tensor mapping so the exporter and the server share
one definition of the engine contract.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib

import tensorrt as trt
import torch
from torch import nn

from openpi.models_pytorch.preprocessing_pytorch import IMAGE_KEYS
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

logger = logging.getLogger("openpi")

ENGINE_INPUT_NAMES = (
    "images",
    "img_masks",
    "lang_tokens",
    "lang_masks",
    "noise",
    "prev_chunk_left_over",
    "inference_delay",
)
ENGINE_OUTPUT_NAME = "actions"
ENGINE_METADATA_FORMAT = "tron2-pi05-tensorrt/1"

_TRT_TO_TORCH_DTYPE = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}


@dataclasses.dataclass(frozen=True)
class EngineTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class EngineObservation:
    """Observation flattened to the engine's input tensors (all on the engine device)."""

    images: torch.Tensor  # float32 [B, 3 * num_views, H, W], scaled to [-1, 1]
    img_masks: torch.Tensor  # bool [B, num_views]
    lang_tokens: torch.Tensor  # int64 [B, max_token_len]
    lang_masks: torch.Tensor  # bool [B, max_token_len]


def flatten_observation(observation) -> EngineObservation:
    """Apply the PyTorch inference preprocessing and stack the camera views in IMAGE_KEYS order."""
    observation = _preprocessing.preprocess_observation_pytorch(observation, train=False)
    if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
        raise ValueError("The TensorRT engine requires a tokenized prompt in the observation")
    return EngineObservation(
        images=torch.cat([observation.images[key] for key in IMAGE_KEYS], dim=1).to(torch.float32),
        img_masks=torch.stack([observation.image_masks[key] for key in IMAGE_KEYS], dim=1).to(torch.bool),
        lang_tokens=observation.tokenized_prompt.to(torch.int64),
        lang_masks=observation.tokenized_prompt_mask.to(torch.bool),
    )


def engine_metadata_path(engine_path: pathlib.Path) -> pathlib.Path:
    return engine_path.with_name(engine_path.name + ".json")


def load_engine_metadata(engine_path: pathlib.Path) -> dict:
    metadata_path = engine_metadata_path(engine_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"TensorRT engine metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != ENGINE_METADATA_FORMAT:
        raise ValueError(f"Unsupported engine metadata format {metadata.get('format')!r} in {metadata_path}")
    return metadata


class TensorRTEngine:
    """Static-shape TensorRT engine executed through persistent torch buffers and a CUDA graph.

    All I/O tensors are allocated once at load; each call copies the inputs into
    those buffers, replays the captured graph (captured lazily on the first call,
    after one warm-up enqueue) and returns a copy of the output buffer.
    """

    def __init__(self, engine_path: pathlib.Path, device: torch.device):
        if device.type != "cuda":
            raise ValueError(f"TensorRT engines run on CUDA devices only, got {device}")
        self.engine_path = engine_path
        self.device = device
        self._logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self._logger, "")
        runtime = trt.Runtime(self._logger)
        with open(engine_path, "rb") as handle:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine {engine_path} (TensorRT {trt.__version__})")
        self._context = self._engine.create_execution_context()

        self.inputs: dict[str, EngineTensorSpec] = {}
        self.outputs: dict[str, EngineTensorSpec] = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise ValueError(f"Engine tensor {name} has a dynamic shape {shape}; the pi05 engine must be static")
            spec = EngineTensorSpec(name=name, shape=shape, dtype=_TRT_TO_TORCH_DTYPE[self._engine.get_tensor_dtype(name)])
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = spec
            else:
                self.outputs[name] = spec

        with torch.cuda.device(device):
            self._buffers = {
                spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=device)
                for spec in (*self.inputs.values(), *self.outputs.values())
            }
        for name, buffer in self._buffers.items():
            self._context.set_tensor_address(name, buffer.data_ptr())
        # TensorRT adds cudaStreamSynchronize calls when enqueued on the legacy
        # default stream, so the engine owns a stream; callers see a synchronous API.
        self._stream = torch.cuda.Stream(device)
        self._graph: torch.cuda.CUDAGraph | None = None

    def describe(self) -> str:
        lines = [f"TensorRT engine {self.engine_path}"]
        for kind, specs in (("input", self.inputs), ("output", self.outputs)):
            for spec in specs.values():
                lines.append(f"  {kind} {spec.name}: {list(spec.shape)} {spec.dtype}")
        return "\n".join(lines)

    def _capture_graph(self) -> torch.cuda.CUDAGraph:
        # One eager enqueue first: lazy allocations inside TensorRT must not
        # happen during capture.
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT warm-up enqueue failed")
        self._stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self._stream):
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT enqueue failed during CUDA graph capture")
        return graph

    def __call__(self, **inputs: torch.Tensor) -> torch.Tensor:
        """Run the engine on the given inputs; returns the output on the caller's stream, already computed."""
        if set(inputs) != set(self.inputs):
            raise ValueError(f"Engine inputs {sorted(self.inputs)} != provided {sorted(inputs)}")
        for name, tensor in inputs.items():
            spec = self.inputs[name]
            if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                raise ValueError(
                    f"Engine input {name} expects {list(spec.shape)} {spec.dtype}, got {list(tensor.shape)} {tensor.dtype}"
                )
        caller_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(caller_stream)
        with torch.cuda.stream(self._stream):
            for name, tensor in inputs.items():
                self._buffers[name].copy_(tensor, non_blocking=True)
            if self._graph is None:
                self._graph = self._capture_graph()
            self._graph.replay()
            output = self._buffers[ENGINE_OUTPUT_NAME].clone()
        self._stream.synchronize()
        return output


class PI0TensorRT(nn.Module):
    """Drop-in replacement for PI0Pytorch that runs sample_actions on a TensorRT engine."""

    def __init__(self, config, engine_path: pathlib.Path | str, device: torch.device | str):
        super().__init__()
        self.config = config
        self.engine_path = pathlib.Path(engine_path)
        self.device = torch.device(device)
        self.metadata = load_engine_metadata(self.engine_path)
        self.num_steps = int(self.metadata["num_steps"])
        self._check_metadata_contract()
        self.engine = TensorRTEngine(self.engine_path, self.device)
        self._check_engine_contract()
        logger.info("%s", self.engine.describe())

    def _check_metadata_contract(self) -> None:
        expected = {
            "action_horizon": int(self.config.action_horizon),
            "action_dim": int(self.config.action_dim),
            "max_token_len": int(self.config.max_token_len),
            "num_views": len(IMAGE_KEYS),
        }
        mismatches = [
            f"{key}: engine {self.metadata.get(key)!r} != config {value!r}"
            for key, value in expected.items()
            if self.metadata.get(key) != value
        ]
        if self.metadata.get("tensorrt_version") != trt.__version__:
            mismatches.append(f"tensorrt_version: engine {self.metadata.get('tensorrt_version')!r} != runtime {trt.__version__!r}")
        if mismatches:
            raise ValueError(f"TensorRT engine {self.engine_path} does not match this policy: " + "; ".join(mismatches))

    def _check_engine_contract(self) -> None:
        if tuple(self.engine.inputs) != ENGINE_INPUT_NAMES or tuple(self.engine.outputs) != (ENGINE_OUTPUT_NAME,):
            raise ValueError(
                f"Engine I/O {tuple(self.engine.inputs)} -> {tuple(self.engine.outputs)} does not match the pi05 contract"
            )
        horizon, action_dim = int(self.config.action_horizon), int(self.config.action_dim)
        batch = self.engine.inputs["noise"].shape[0]
        expected_shapes = {
            "images": (batch, 3 * len(IMAGE_KEYS), *_preprocessing.IMAGE_RESOLUTION),
            "img_masks": (batch, len(IMAGE_KEYS)),
            "lang_tokens": (batch, int(self.config.max_token_len)),
            "lang_masks": (batch, int(self.config.max_token_len)),
            "noise": (batch, horizon, action_dim),
            "prev_chunk_left_over": (batch, horizon, action_dim),
            "inference_delay": (1,),
        }
        for name, shape in expected_shapes.items():
            if self.engine.inputs[name].shape != shape:
                raise ValueError(f"Engine input {name} has shape {self.engine.inputs[name].shape}, expected {shape}")
        if self.engine.outputs[ENGINE_OUTPUT_NAME].shape != (batch, horizon, action_dim):
            raise ValueError(f"Engine output shape {self.engine.outputs[ENGINE_OUTPUT_NAME].shape} != {(batch, horizon, action_dim)}")

    @property
    def action_horizon(self) -> int:
        return self.config.action_horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def batch_size(self) -> int:
        return self.engine.inputs["noise"].shape[0]

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    @torch.no_grad()
    def sample_actions(
        self,
        device,
        observation,
        noise=None,
        num_steps=None,
        *,
        prev_chunk_left_over=None,
        prev_chunk_left_over_len=None,
        inference_delay=0,
        prefix_horizon=None,
        max_guidance_weight=None,
        trained_rtc_mode=False,
    ) -> torch.Tensor:
        """Mirror of PI0Pytorch.sample_actions on the engine.

        ``num_steps`` is baked into the engine; ``prefix_horizon`` and
        ``max_guidance_weight`` only parameterize the VJP guidance that the
        PyTorch model does not implement either, so they are accepted and unused.
        """
        if num_steps is not None and int(num_steps) != self.num_steps:
            raise ValueError(f"The engine was exported with num_steps={self.num_steps}; got {num_steps}")
        if prev_chunk_left_over is not None and not trained_rtc_mode:
            raise NotImplementedError(
                "Inference-time RTC guidance (VJP) is not implemented for the TensorRT model; "
                "use trained_rtc_mode=True with a checkpoint trained with rtc_training_simulated_delay."
            )
        flat = flatten_observation(observation)
        bsize = flat.images.shape[0]
        if bsize != self.batch_size:
            raise ValueError(f"The engine is built for batch {self.batch_size}, got {bsize}")
        horizon, action_dim = int(self.config.action_horizon), int(self.config.action_dim)
        actions_shape = (bsize, horizon, action_dim)

        if noise is None:
            noise = self.sample_noise(actions_shape, self.device)
        noise = noise.to(device=self.device, dtype=torch.float32)

        if prev_chunk_left_over is None:
            prev_chunk = torch.zeros(actions_shape, dtype=torch.float32, device=self.device)
            delay = 0
        else:
            prev_chunk_left_over = prev_chunk_left_over.to(device=self.device, dtype=torch.float32)
            if prev_chunk_left_over.shape[1] < horizon:
                prev_chunk = torch.zeros(actions_shape, dtype=torch.float32, device=self.device)
                prev_chunk[:, : prev_chunk_left_over.shape[1], :] = prev_chunk_left_over
            else:
                prev_chunk = prev_chunk_left_over
            delay = int(inference_delay)
        if not 0 <= delay <= horizon:
            raise ValueError(f"inference_delay must be within [0, {horizon}], got {delay}")

        # The engine call is synchronous, so Policy.infer's infer_ms (measured
        # before the result moves to the host) covers the whole inference.
        return self.engine(
            images=flat.images.to(self.device),
            img_masks=flat.img_masks.to(self.device),
            lang_tokens=flat.lang_tokens.to(self.device),
            lang_masks=flat.lang_masks.to(self.device),
            noise=noise,
            prev_chunk_left_over=prev_chunk,
            inference_delay=torch.tensor([delay], dtype=torch.int64, device=self.device),
        )
