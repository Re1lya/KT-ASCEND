# SPDX-License-Identifier: Apache-2.0

"""Single-layer CPU/accelerator Hybrid MoE coordination primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


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

