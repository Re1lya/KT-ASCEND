# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import FixedExpertPlacement, TorchNPUExpertProvider


FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_hybrid_npu", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


def _reference_expert(hidden_states, weight, expert_weight=1.0):
    rows = []
    for hidden in hidden_states:
        x = hidden.float()
        gate = weight["gate"] @ x
        up = weight["up"] @ x
        result = weight["down"] @ (torch.nn.functional.silu(gate) * up)
        rows.append((result * expert_weight).to(torch.bfloat16))
    return torch.stack(rows)


@pytest.mark.parametrize("logical_id", [1, 3])
def test_single_npu_expert_matches_float_reference(logical_id):
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    placement = FixedExpertPlacement(4, torch.tensor([False, True, False, True]))
    provider = TorchNPUExpertProvider(placement, weights)
    torch.manual_seed(tiny_fixture.SEED + logical_id)
    hidden_cpu = torch.randn(3, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    hidden_npu = hidden_cpu.to("npu")
    other_id = 0 if logical_id == 1 else 2
    ids = torch.tensor([[logical_id, other_id]] * 3, dtype=torch.int64, device="npu")
    routing_weights = torch.tensor([[1.0, 0.0]] * 3, dtype=torch.float32, device="npu")

    actual = provider.forward(hidden_npu, ids, routing_weights)
    torch.npu.current_stream().synchronize()
    physical_weight = {name: tensor[logical_id] for name, tensor in weights.items()}
    expected = _reference_expert(hidden_cpu, physical_weight)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-3)


def test_cpu_owned_routes_have_exact_zero_npu_contribution():
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    placement = FixedExpertPlacement(4, torch.tensor([False, True, False, True]))
    provider = TorchNPUExpertProvider(placement, weights)
    hidden = torch.randn(4, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16, device="npu")
    ids = torch.tensor([[0, 2]] * 4, dtype=torch.int64, device="npu")
    routing_weights = torch.tensor([[0.4, 0.6]] * 4, dtype=torch.float32, device="npu")
    actual = provider.forward(hidden, ids, routing_weights)
    torch.npu.current_stream().synchronize()
    assert torch.count_nonzero(actual).item() == 0


def test_physical_weights_are_reordered_to_global_logical_ids():
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    placement = FixedExpertPlacement(4, torch.tensor([False, True, False, True]), mapping)
    provider = TorchNPUExpertProvider(placement, weights)
    hidden_cpu = torch.randn(1, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    ids = torch.tensor([[1, 0]], dtype=torch.int64, device="npu")
    routing_weights = torch.tensor([[1.0, 0.0]], dtype=torch.float32, device="npu")

    actual = provider.forward(hidden_cpu.to("npu"), ids, routing_weights)
    torch.npu.current_stream().synchronize()
    logical_to_physical = torch.argsort(mapping)
    physical_id = int(logical_to_physical[1])
    physical_weight = {name: tensor[physical_id] for name, tensor in weights.items()}
    expected = _reference_expert(hidden_cpu, physical_weight)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-3)
