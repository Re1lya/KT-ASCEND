# SPDX-License-Identifier: Apache-2.0

"""Single-layer CPU/accelerator Hybrid MoE coordination primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FixedExpertPlacement:
    """Validated fixed ownership and physical/logical mapping for one MoE layer.

    ``accelerator_mask[logical_id]`` is true for an accelerator-owned expert
    and false for a CPU-owned expert. ``physical_to_logical_map[physical_id]``
    identifies the global logical expert loaded from that physical weight slot.
    """

    num_experts: int
    accelerator_mask: torch.Tensor
    physical_to_logical_map: torch.Tensor

    def __init__(
        self,
        num_experts: int,
        accelerator_mask: torch.Tensor,
        physical_to_logical_map: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if not isinstance(accelerator_mask, torch.Tensor):
            raise TypeError("accelerator_mask must be a torch.Tensor")
        if accelerator_mask.device.type != "cpu":
            raise ValueError("accelerator_mask must be a CPU tensor")
        if accelerator_mask.dtype != torch.bool:
            raise ValueError(f"accelerator_mask must use torch.bool, got {accelerator_mask.dtype}")
        if accelerator_mask.ndim != 1 or accelerator_mask.numel() != num_experts:
            raise ValueError(
                f"accelerator_mask must have shape [{num_experts}], got {list(accelerator_mask.shape)}"
            )

        if physical_to_logical_map is None:
            mapping = torch.arange(num_experts, dtype=torch.int32)
        elif isinstance(physical_to_logical_map, torch.Tensor):
            if physical_to_logical_map.device.type != "cpu":
                raise ValueError("physical_to_logical_map must be a CPU tensor")
            if physical_to_logical_map.dtype not in (torch.int32, torch.int64):
                raise ValueError("physical_to_logical_map must use torch.int32 or torch.int64")
            mapping = physical_to_logical_map.contiguous().view(-1).to(torch.int32)
        else:
            mapping = torch.tensor(list(physical_to_logical_map), dtype=torch.int32)

        if mapping.numel() != num_experts:
            raise ValueError(
                f"physical_to_logical_map must contain {num_experts} entries, got {mapping.numel()}"
            )
        if torch.sort(mapping).values.tolist() != list(range(num_experts)):
            raise ValueError("physical_to_logical_map must be a permutation of all global expert IDs")

        object.__setattr__(self, "num_experts", int(num_experts))
        object.__setattr__(self, "accelerator_mask", accelerator_mask.contiguous().clone())
        object.__setattr__(self, "physical_to_logical_map", mapping.clone())

    @property
    def cpu_experts(self) -> tuple[int, ...]:
        return tuple(index for index, on_accelerator in enumerate(self.accelerator_mask.tolist()) if not on_accelerator)

    @property
    def accelerator_experts(self) -> tuple[int, ...]:
        return tuple(index for index, on_accelerator in enumerate(self.accelerator_mask.tolist()) if on_accelerator)

    def validate_routes(self, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> None:
        if expert_ids.shape != routing_weights.shape:
            raise ValueError(
                "expert_ids and routing_weights must have the same shape, got "
                f"{list(expert_ids.shape)} and {list(routing_weights.shape)}"
            )
        if expert_ids.ndim != 2:
            raise ValueError(f"expert_ids must be rank 2 [tokens, top_k], got rank {expert_ids.ndim}")
        if expert_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("expert_ids must use torch.int32 or torch.int64")
        if not torch.is_floating_point(routing_weights):
            raise ValueError("routing_weights must use a floating-point dtype")
        if expert_ids.numel() == 0:
            raise ValueError("at least one routed expert is required")
        minimum = int(expert_ids.min().item())
        maximum = int(expert_ids.max().item())
        if minimum < 0 or maximum >= self.num_experts:
            raise ValueError(
                f"global expert IDs must be in [0, {self.num_experts}), observed [{minimum}, {maximum}]"
            )
        if not bool(torch.isfinite(routing_weights).all().item()):
            raise ValueError("routing_weights must contain only finite values")

    def cpu_route_ids(self, expert_ids: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """Keep global CPU expert IDs and replace accelerator experts with -1."""
        self.validate_routes(expert_ids, routing_weights)
        mask = self.accelerator_mask.to(expert_ids.device)[expert_ids.to(torch.long)]
        return expert_ids.masked_fill(mask, -1)

    def accelerator_route_weights(
        self, expert_ids: torch.Tensor, routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """Keep weights for accelerator experts and zero CPU expert weights."""
        self.validate_routes(expert_ids, routing_weights)
        mask = self.accelerator_mask.to(expert_ids.device)[expert_ids.to(torch.long)]
        return routing_weights * mask.to(routing_weights.dtype)


class TorchNPUExpertProvider:
    """Plain-torch NPU implementation for a fixed single-layer expert fixture.

    Source tensors are indexed by physical weight slot. They are reordered once
    at construction so the resident tensors are indexed by global logical ID.
    """

    def __init__(
        self,
        placement: FixedExpertPlacement,
        weights: Mapping[str, torch.Tensor],
        *,
        device: str | torch.device = "npu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        required = {"gate", "up", "down"}
        if set(weights) != required:
            raise ValueError(f"weights must contain exactly {sorted(required)}, got {sorted(weights)}")

        gate = weights["gate"]
        up = weights["up"]
        down = weights["down"]
        if gate.ndim != 3 or up.ndim != 3 or down.ndim != 3:
            raise ValueError("gate, up, and down weights must be rank-3 expert tensors")
        if gate.shape != up.shape:
            raise ValueError(f"gate and up shapes must match, got {list(gate.shape)} and {list(up.shape)}")
        experts, intermediate_size, hidden_size = gate.shape
        if experts != placement.num_experts:
            raise ValueError(f"weight expert dimension must be {placement.num_experts}, got {experts}")
        if down.shape != (experts, hidden_size, intermediate_size):
            raise ValueError(
                "down weights must have shape "
                f"[{experts}, {hidden_size}, {intermediate_size}], got {list(down.shape)}"
            )
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(f"unsupported NPU expert dtype: {dtype}")

        logical_to_physical = torch.argsort(placement.physical_to_logical_map.to(torch.int64))
        self.placement = placement
        requested_device = torch.device(device)
        self.dtype = dtype
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.gate = gate.index_select(0, logical_to_physical).to(requested_device, dtype=dtype).contiguous()
        self.up = up.index_select(0, logical_to_physical).to(requested_device, dtype=dtype).contiguous()
        self.down = down.index_select(0, logical_to_physical).to(requested_device, dtype=dtype).contiguous()
        self.device = self.gate.device

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return only accelerator-owned routed contributions on the NPU."""
        self.placement.validate_routes(expert_ids, routing_weights)
        if hidden_states.device != self.device:
            raise ValueError(f"hidden_states must be on {self.device}, got {hidden_states.device}")
        if hidden_states.dtype != self.dtype:
            raise ValueError(f"hidden_states must use {self.dtype}, got {hidden_states.dtype}")

        original_shape = hidden_states.shape
        flat_hidden = hidden_states.contiguous().view(-1, self.hidden_size)
        ids = expert_ids.to(self.device, dtype=torch.int64).contiguous().view(flat_hidden.shape[0], -1)
        weights = routing_weights.to(self.device, dtype=torch.float32).contiguous().view_as(ids)
        if ids.shape[0] != flat_hidden.shape[0]:
            raise ValueError("hidden_states and routes must describe the same flattened token count")

        output = torch.zeros(
            (flat_hidden.shape[0], self.hidden_size),
            dtype=torch.float32,
            device=self.device,
        )
        accelerator_weights = self.placement.accelerator_route_weights(ids, weights)
        for logical_id in self.placement.accelerator_experts:
            selected_routes = torch.nonzero(ids == logical_id, as_tuple=False)
            token_indices = selected_routes[:, 0]
            route_indices = selected_routes[:, 1]
            selected_hidden = flat_hidden.index_select(0, token_indices)
            gate = F.linear(selected_hidden, self.gate[logical_id])
            up = F.linear(selected_hidden, self.up[logical_id])
            expert_output = F.linear(F.silu(gate) * up, self.down[logical_id])
            selected_weights = accelerator_weights[token_indices, route_indices].unsqueeze(1)
            output.index_add_(0, token_indices, expert_output.float() * selected_weights)

        return output.to(hidden_states.dtype).view(original_shape)


@dataclass(frozen=True)
class HybridMoEResult:
    """Separate contributions and their additive merged output."""

    output: torch.Tensor
    cpu_contribution: torch.Tensor
    accelerator_contribution: torch.Tensor


class HybridMoECoordinator:
    """Coordinate one fixed-placement CPU/NPU routed-MoE layer."""

    def __init__(
        self,
        cpu_wrapper,
        accelerator_provider: TorchNPUExpertProvider,
        placement: FixedExpertPlacement,
    ) -> None:
        if int(cpu_wrapper.num_experts) != placement.num_experts:
            raise ValueError(
                f"CPU wrapper has {cpu_wrapper.num_experts} experts, placement has {placement.num_experts}"
            )
        wrapper_mask = cpu_wrapper.gpu_experts_mask.to(device="cpu", dtype=torch.bool).view(-1)
        if not torch.equal(wrapper_mask, placement.accelerator_mask):
            raise ValueError("CPU wrapper gpu_experts_mask does not match the fixed accelerator placement")
        if int(cpu_wrapper.max_deferred_experts_per_token) != 0:
            raise ValueError("HybridMoECoordinator requires max_deferred_experts_per_token=0")
        provider_placement = accelerator_provider.placement
        if not torch.equal(provider_placement.accelerator_mask, placement.accelerator_mask) or not torch.equal(
            provider_placement.physical_to_logical_map, placement.physical_to_logical_map
        ):
            raise ValueError("accelerator provider placement does not match coordinator placement")
        if int(cpu_wrapper.hidden_size) != accelerator_provider.hidden_size:
            raise ValueError("CPU and accelerator expert hidden sizes do not match")

        self.cpu_wrapper = cpu_wrapper
        self.accelerator_provider = accelerator_provider
        self.placement = placement

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> None:
        self.placement.validate_routes(expert_ids, routing_weights)
        if hidden_states.device != self.accelerator_provider.device:
            raise ValueError(
                f"hidden_states must be on {self.accelerator_provider.device}, got {hidden_states.device}"
            )
        if hidden_states.dtype != self.accelerator_provider.dtype:
            raise ValueError(
                f"hidden_states must use {self.accelerator_provider.dtype}, got {hidden_states.dtype}"
            )
        token_count = hidden_states.numel() // hidden_states.shape[-1]
        if hidden_states.shape[-1] != self.accelerator_provider.hidden_size:
            raise ValueError(
                f"hidden_states last dimension must be {self.accelerator_provider.hidden_size}, "
                f"got {hidden_states.shape[-1]}"
            )
        if expert_ids.shape[0] != token_count:
            raise ValueError("hidden_states and routes must describe the same flattened token count")

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "npu":
            torch.npu.current_stream(device).synchronize()
        elif device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        else:
            raise ValueError(f"unsupported accelerator device: {device}")

    def forward_sequential(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> HybridMoEResult:
        """Run CPU then NPU with explicit synchronization before the merge."""
        self._validate_inputs(hidden_states, expert_ids, routing_weights)
        hidden_cpu = hidden_states.to(device="cpu", dtype=torch.bfloat16)
        ids_cpu = expert_ids.to(device="cpu", dtype=torch.int64)
        weights_cpu = routing_weights.to(device="cpu", dtype=torch.float32)
        cpu_host = self.cpu_wrapper.forward(hidden_cpu, ids_cpu, weights_cpu)

        accelerator_output = self.accelerator_provider.forward(hidden_states, expert_ids, routing_weights)
        self._synchronize(hidden_states.device)
        cpu_output = cpu_host.to(hidden_states.device, non_blocking=False)
        output = cpu_output + accelerator_output
        self._synchronize(hidden_states.device)
        return HybridMoEResult(output, cpu_output, accelerator_output)

    def forward_overlapped(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        routing_weights: torch.Tensor,
        device_stream_handle: int,
    ) -> HybridMoEResult:
        """Overlap CPUInfer with NPU experts, then merge on the NPU stream."""
        self._validate_inputs(hidden_states, expert_ids, routing_weights)
        if not isinstance(device_stream_handle, int) or device_stream_handle == 0:
            raise ValueError("device_stream_handle must be a non-zero integer")

        self.cpu_wrapper.submit_forward(hidden_states, expert_ids, routing_weights, device_stream_handle)
        accelerator_output = self.accelerator_provider.forward(hidden_states, expert_ids, routing_weights)
        cpu_output = self.cpu_wrapper.sync_forward(hidden_states, device_stream_handle).clone()
        output = cpu_output + accelerator_output
        return HybridMoEResult(output, cpu_output, accelerator_output)
