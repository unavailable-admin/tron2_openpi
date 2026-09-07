"""Deployment-only fast-weight simulation for the PI0 action expert.

This module intentionally provides no training or policy-serving integration.
It measures the compute and memory cost of a deterministic, untrained TTT
layer while the checkpoint's slow weights remain frozen.
"""

from __future__ import annotations

import dataclasses
import enum
import math

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class TTTUpdateMode(enum.StrEnum):
    APPLY_ONLY = "apply_only"
    UPDATE_ONCE = "update_once"
    EVERY_PASS = "every_pass"


@dataclasses.dataclass(frozen=True)
class TTTConfig:
    width: int
    fast_hidden_size: int
    layer_count: int
    register_tokens: int = 16
    inner_learning_rate: float = 0.1
    gate_init: float = 0.001
    initialization_seed: int = 20260907

    @classmethod
    def from_action_expert_config(cls, action_expert_config: object) -> "TTTConfig":
        return cls(
            width=int(action_expert_config.width),
            fast_hidden_size=int(action_expert_config.mlp_dim),
            layer_count=int(action_expert_config.depth),
        )

    def validate(self) -> None:
        if self.width <= 0 or self.fast_hidden_size <= 0 or self.layer_count <= 0:
            raise ValueError("TTT dimensions and layer count must be positive")
        if self.register_tokens < 0:
            raise ValueError("TTT register token count must be non-negative")
        if self.inner_learning_rate <= 0:
            raise ValueError("TTT inner learning rate must be positive")


@dataclasses.dataclass(frozen=True)
class FastWeightLayerState:
    first_weight: torch.Tensor
    first_bias: torch.Tensor
    second_weight: torch.Tensor
    second_bias: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.first_weight, self.first_bias, self.second_weight, self.second_bias)

    def detached_clone(self) -> "FastWeightLayerState":
        return FastWeightLayerState(*(tensor.detach().clone() for tensor in self.tensors()))


@dataclasses.dataclass
class TTTTiming:
    update_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = dataclasses.field(default_factory=list)
    apply_event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = dataclasses.field(default_factory=list)
    update_count: int = 0
    apply_count: int = 0

    def elapsed_ms(self) -> dict[str, float]:
        return {
            "update_cuda_ms": float(sum(start.elapsed_time(end) for start, end in self.update_event_pairs)),
            "apply_cuda_ms": float(sum(start.elapsed_time(end) for start, end in self.apply_event_pairs)),
        }


class TTTFastWeightLayer(nn.Module):
    def __init__(self, config: TTTConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.key_projection = nn.Linear(config.width, config.width, bias=False)
        self.value_projection = nn.Linear(config.width, config.width, bias=False)
        self.initial_first_weight = nn.Parameter(torch.empty(config.fast_hidden_size, config.width), requires_grad=False)
        self.initial_first_bias = nn.Parameter(torch.empty(config.fast_hidden_size), requires_grad=False)
        self.initial_second_weight = nn.Parameter(
            torch.empty(config.width, config.fast_hidden_size), requires_grad=False
        )
        self.initial_second_bias = nn.Parameter(torch.empty(config.width), requires_grad=False)
        self.register_embeddings = nn.Parameter(
            torch.empty(config.register_tokens, config.width), requires_grad=False
        )
        self.gate = nn.Parameter(torch.full((config.width,), config.gate_init), requires_grad=False)
        self._initialize_deterministically()

    def _initialize_deterministically(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.initialization_seed + self.layer_index)
        projection_bound = 1.0 / math.sqrt(self.config.width)
        fast_first_bound = 1.0 / math.sqrt(self.config.width)
        for projection in (self.query_projection, self.key_projection, self.value_projection):
            nn.init.uniform_(projection.weight, -projection_bound, projection_bound, generator=generator)
            projection.weight.requires_grad_(False)
        nn.init.uniform_(
            self.initial_first_weight, -fast_first_bound, fast_first_bound, generator=generator
        )
        nn.init.zeros_(self.initial_first_bias)
        # A zero output projection gives the untrained branch an exact zero
        # residual before its first inner update and avoids arbitrary recurrent
        # amplification in the conservative every-pass-update simulation.
        nn.init.zeros_(self.initial_second_weight)
        nn.init.zeros_(self.initial_second_bias)
        if self.config.register_tokens:
            nn.init.normal_(
                self.register_embeddings,
                mean=0.0,
                std=1.0 / math.sqrt(self.config.width),
                generator=generator,
            )

    def initial_state(self) -> FastWeightLayerState:
        return FastWeightLayerState(
            self.initial_first_weight.detach().clone(),
            self.initial_first_bias.detach().clone(),
            self.initial_second_weight.detach().clone(),
            self.initial_second_bias.detach().clone(),
        )

    @staticmethod
    def _fast_forward(tokens: torch.Tensor, state: FastWeightLayerState) -> torch.Tensor:
        hidden = F.linear(tokens, state.first_weight, state.first_bias)
        hidden = F.gelu(hidden)
        return F.linear(hidden, state.second_weight, state.second_bias)

    def _tokens_with_registers(self, hidden_states: torch.Tensor, use_register_tokens: bool) -> torch.Tensor:
        if not use_register_tokens or self.config.register_tokens == 0:
            return hidden_states
        registers = self.register_embeddings.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
        return torch.cat((hidden_states, registers), dim=1)

    def update(
        self,
        hidden_states: torch.Tensor,
        state: FastWeightLayerState,
        use_register_tokens: bool,
    ) -> FastWeightLayerState:
        tokens = self._tokens_with_registers(hidden_states, use_register_tokens)
        with torch.no_grad():
            keys = self.key_projection(tokens).detach()
            values = self.value_projection(tokens).detach()
        differentiable_state = FastWeightLayerState(
            *(tensor.detach().requires_grad_(True) for tensor in state.tensors())
        )
        with torch.enable_grad():
            predictions = self._fast_forward(keys, differentiable_state)
            loss = F.mse_loss(predictions.float(), values.float())
            gradients = torch.autograd.grad(loss, differentiable_state.tensors(), create_graph=False)
        updated_tensors = tuple(
            (parameter - self.config.inner_learning_rate * gradient).detach()
            for parameter, gradient in zip(differentiable_state.tensors(), gradients, strict=True)
        )
        return FastWeightLayerState(*updated_tensors)

    def apply(
        self,
        hidden_states: torch.Tensor,
        state: FastWeightLayerState,
        use_register_tokens: bool,
        gate_zero: bool,
    ) -> torch.Tensor:
        tokens = self._tokens_with_registers(hidden_states, use_register_tokens)
        queries = self.query_projection(tokens)
        fast_output = self._fast_forward(queries, state)[:, : hidden_states.shape[1]]
        gate = torch.zeros_like(self.gate) if gate_zero else self.gate
        output = hidden_states + torch.tanh(gate) * fast_output
        return output.detach()


class TTTFastWeightStack(nn.Module):
    def __init__(self, config: TTTConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.layers = nn.ModuleList(TTTFastWeightLayer(config, index) for index in range(config.layer_count))

    def initial_states(self) -> list[FastWeightLayerState]:
        return [layer.initial_state() for layer in self.layers]

    def effective_config(self) -> dict[str, int | float]:
        fast_parameters_per_layer = (
            self.config.fast_hidden_size * self.config.width
            + self.config.fast_hidden_size
            + self.config.width * self.config.fast_hidden_size
            + self.config.width
        )
        return {
            **dataclasses.asdict(self.config),
            "fast_parameters_per_layer": fast_parameters_per_layer,
            "fast_parameters_total": fast_parameters_per_layer * self.config.layer_count,
        }


class TTTSession:
    def __init__(self, stack: TTTFastWeightStack):
        self.stack = stack
        self.states = stack.initial_states()
        self.last_updated_observation_id: str | None = None

    def reset(self) -> None:
        self.states = self.stack.initial_states()
        self.last_updated_observation_id = None

    def all_finite(self) -> bool:
        return all(torch.isfinite(tensor).all().item() for state in self.states for tensor in state.tensors())

    def start_request(
        self,
        observation_id: str,
        mode: TTTUpdateMode,
        layer_limit: int,
        *,
        use_register_tokens: bool,
        gate_zero: bool = False,
        record_cuda_timing: bool = False,
    ) -> "TTTRequest":
        if not 0 <= layer_limit <= self.stack.config.layer_count:
            raise ValueError(f"layer_limit must be in [0, {self.stack.config.layer_count}], got {layer_limit}")
        duplicate_observation = observation_id == self.last_updated_observation_id
        return TTTRequest(
            session=self,
            observation_id=observation_id,
            mode=mode,
            layer_limit=layer_limit,
            use_register_tokens=use_register_tokens,
            gate_zero=gate_zero,
            suppress_update=duplicate_observation,
            record_cuda_timing=record_cuda_timing,
        )


class TTTRequest:
    def __init__(
        self,
        *,
        session: TTTSession,
        observation_id: str,
        mode: TTTUpdateMode,
        layer_limit: int,
        use_register_tokens: bool,
        gate_zero: bool,
        suppress_update: bool,
        record_cuda_timing: bool,
    ):
        self.session = session
        self.observation_id = observation_id
        self.mode = mode
        self.layer_limit = layer_limit
        self.use_register_tokens = use_register_tokens
        self.gate_zero = gate_zero
        self.suppress_update = suppress_update
        self.record_cuda_timing = record_cuda_timing
        self.denoise_pass_index = 0
        self.timing = TTTTiming()

    def begin_denoise_pass(self, pass_index: int) -> None:
        self.denoise_pass_index = pass_index

    def _should_update(self) -> bool:
        if self.suppress_update or self.mode == TTTUpdateMode.APPLY_ONLY:
            return False
        if self.mode == TTTUpdateMode.UPDATE_ONCE:
            return self.denoise_pass_index == 0
        return True

    def _event_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if not self.record_cuda_timing:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    @staticmethod
    def _finish_event(event_pair: tuple[torch.cuda.Event, torch.cuda.Event] | None) -> None:
        if event_pair is not None:
            event_pair[1].record()

    def process_layer(self, layer_index: int, hidden_states: torch.Tensor) -> torch.Tensor:
        if layer_index >= self.layer_limit:
            return hidden_states
        layer = self.session.stack.layers[layer_index]
        if self._should_update():
            update_events = self._event_pair()
            new_state = layer.update(hidden_states, self.session.states[layer_index], self.use_register_tokens)
            self._finish_event(update_events)
            if update_events is not None:
                self.timing.update_event_pairs.append(update_events)
            self.session.states[layer_index] = new_state
            self.timing.update_count += 1

        apply_events = self._event_pair()
        output = layer.apply(
            hidden_states,
            self.session.states[layer_index],
            self.use_register_tokens,
            self.gate_zero,
        )
        self._finish_event(apply_events)
        if apply_events is not None:
            self.timing.apply_event_pairs.append(apply_events)
        self.timing.apply_count += 1
        return output

    def finish(self) -> None:
        if self.timing.update_count:
            self.session.last_updated_observation_id = self.observation_id
