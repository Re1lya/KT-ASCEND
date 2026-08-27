# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

import kt_kernel
from kt_kernel import KTMoEWrapper

FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_routed_fixture", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


pytestmark = pytest.mark.skipif(
    kt_kernel.__cpu_variant__ != "arm",
    reason="LLAMAFILE routed correctness requires the A3 ARM extension",
)


def _make_wrapper(path: Path, *, num_experts: int, layer_idx: int = 0):
    return KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=num_experts,
        num_experts_per_tok=2,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        moe_intermediate_size=256,
        gpu_experts_mask=None,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=str(path),
        chunked_prefill_size=64,
        method="LLAMAFILE",
        numa_nodes=[0],
    )


def _reference(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    physical_to_logical_map: torch.Tensor,
) -> torch.Tensor:
    logical_to_physical = torch.empty_like(physical_to_logical_map)
    for physical_id, logical_id in enumerate(physical_to_logical_map.tolist()):
        logical_to_physical[logical_id] = physical_id

    rows = []
    for token_idx in range(hidden_states.shape[0]):
        x = hidden_states[token_idx].float()
        result = torch.zeros_like(x)
        for route_idx in range(expert_ids.shape[1]):
            logical_id = int(expert_ids[token_idx, route_idx])
            physical_id = int(logical_to_physical[logical_id])
            gate = layer_weights["gate"][physical_id] @ x
            up = layer_weights["up"][physical_id] @ x
            expert_output = layer_weights["down"][physical_id] @ (torch.nn.functional.silu(gate) * up)
            result += float(routing_weights[token_idx, route_idx]) * expert_output
        rows.append(result.to(torch.bfloat16))
    return torch.stack(rows)


def _assert_numerical(actual: torch.Tensor, expected: torch.Tensor) -> None:
    difference = actual.float() - expected.float()
    reference_norm = float(torch.linalg.vector_norm(expected.float()))
    relative_l2 = float(torch.linalg.vector_norm(difference)) / reference_norm if reference_norm else 0.0
    max_abs = float(difference.abs().max())
    mean_abs = float(difference.abs().mean())
    print(
        {
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "relative_l2_error": relative_l2,
        }
    )
    assert torch.isfinite(actual).all()
    # Both paths return BF16. Different GEMM and accumulation orders may round
    # by a small number of BF16 ULPs, while mapping/layout bugs are orders of
    # magnitude larger for this deterministic fixture.
    assert max_abs <= 1e-3
    assert mean_abs <= 1e-4
    assert relative_l2 <= 1e-2


@pytest.mark.parametrize(
    ("num_experts", "mapping"),
    [
        (4, [2, 0, 3, 1]),
        (8, [2, 0, 3, 1, 7, 5, 4, 6]),
    ],
)
def test_top2_routing_and_physical_to_logical_mapping(tmp_path, num_experts, mapping):
    fixture_path, _ = tiny_fixture.create_fixture(tmp_path / f"tiny-{num_experts}.gguf", num_experts=num_experts)
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=num_experts)
    physical_to_logical_map = torch.tensor(mapping, dtype=torch.int32)
    wrapper = _make_wrapper(fixture_path, num_experts=num_experts)
    wrapper.load_weights(physical_to_logical_map)

    torch.manual_seed(tiny_fixture.SEED + num_experts)
    hidden_states = torch.randn(1, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    expert_ids = torch.tensor([[1, 3]], dtype=torch.int64)
    routing_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    actual = wrapper.forward(hidden_states, expert_ids, routing_weights)
    expected = _reference(hidden_states, expert_ids, routing_weights, weights[0], physical_to_logical_map)

    _assert_numerical(actual, expected)


@pytest.mark.parametrize("qlen", [1, 2, 8, 32, 64])
def test_decode_and_prefill_token_layout(tmp_path, qlen):
    fixture_path, _ = tiny_fixture.create_fixture(tmp_path / "tiny-prefill.gguf", num_experts=4)
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)
    mapping = torch.arange(4, dtype=torch.int32)
    wrapper = _make_wrapper(fixture_path, num_experts=4)
    wrapper.load_weights(mapping)

    torch.manual_seed(tiny_fixture.SEED + qlen)
    hidden_states = torch.randn(qlen, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    first_ids = torch.arange(qlen, dtype=torch.int64).remainder(4)
    second_ids = (first_ids + 2).remainder(4)
    expert_ids = torch.stack((first_ids, second_ids), dim=1)
    routing_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32).repeat(qlen, 1)
    actual = wrapper.forward(hidden_states, expert_ids, routing_weights)
    expected = _reference(hidden_states, expert_ids, routing_weights, weights[0], mapping)

    assert actual.shape == (qlen, tiny_fixture.DEFAULT_HIDDEN_SIZE)
    _assert_numerical(actual, expected)


def test_router_weight_edges_and_expert_selection(tmp_path):
    fixture_path, _ = tiny_fixture.create_fixture(tmp_path / "tiny-edges.gguf", num_experts=4)
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)
    mapping = torch.arange(4, dtype=torch.int32)
    wrapper = _make_wrapper(fixture_path, num_experts=4)
    wrapper.load_weights(mapping)

    torch.manual_seed(tiny_fixture.SEED + 99)
    hidden_states = torch.randn(2, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    expert_ids = torch.tensor([[1, 3], [1, 2]], dtype=torch.int64)
    routing_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    actual = wrapper.forward(hidden_states, expert_ids, routing_weights)
    expected = _reference(hidden_states, expert_ids, routing_weights, weights[0], mapping)

    _assert_numerical(actual, expected)
