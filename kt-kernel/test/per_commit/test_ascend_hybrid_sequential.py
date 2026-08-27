# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import (
    FixedExpertPlacement,
    HybridMoECoordinator,
    KTMoEWrapper,
    TorchNPUExpertProvider,
)
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_hybrid_sequential", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


def _coordinator(tmp_path, mapping=None):
    fixture, _ = tiny_fixture.create_fixture(tmp_path / "hybrid.gguf", num_experts=4)
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    mapping = torch.arange(4, dtype=torch.int32) if mapping is None else torch.tensor(mapping, dtype=torch.int32)
    accelerator_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
    placement = FixedExpertPlacement(4, accelerator_mask, mapping)
    LlamafileMoEWrapper._gguf_loader_instance = None
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        moe_intermediate_size=tiny_fixture.DEFAULT_INTERMEDIATE_SIZE,
        gpu_experts_mask=accelerator_mask,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=str(fixture),
        chunked_prefill_size=64,
        max_deferred_experts_per_token=0,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    wrapper.load_weights(mapping)
    provider = TorchNPUExpertProvider(placement, weights)
    return HybridMoECoordinator(wrapper, provider, placement), weights, mapping


def _reference(hidden_states, expert_ids, routing_weights, weights, mapping, owned=None):
    logical_to_physical = torch.argsort(mapping.to(torch.int64))
    rows = []
    for token_index, hidden in enumerate(hidden_states):
        x = hidden.float()
        result = torch.zeros_like(x)
        for route_index in range(expert_ids.shape[1]):
            logical_id = int(expert_ids[token_index, route_index])
            if owned is not None and logical_id not in owned:
                continue
            physical_id = int(logical_to_physical[logical_id])
            gate = weights["gate"][physical_id] @ x
            up = weights["up"][physical_id] @ x
            expert_output = weights["down"][physical_id] @ (torch.nn.functional.silu(gate) * up)
            result += float(routing_weights[token_index, route_index]) * expert_output
        rows.append(result.to(torch.bfloat16))
    return torch.stack(rows)


def _assert_numerical(actual, expected):
    difference = actual.float() - expected.float()
    norm = float(torch.linalg.vector_norm(expected.float()))
    metrics = {
        "max_abs_error": float(difference.abs().max()),
        "mean_abs_error": float(difference.abs().mean()),
        "relative_l2_error": float(torch.linalg.vector_norm(difference)) / norm if norm else 0.0,
    }
    print(json.dumps(metrics, sort_keys=True))
    assert metrics["max_abs_error"] <= 2e-3
    assert metrics["mean_abs_error"] <= 2e-4
    assert metrics["relative_l2_error"] <= 2e-2


@pytest.mark.parametrize(
    ("expert_ids", "routing_weights", "zero_side"),
    [
        ([0, 2], [0.4, 0.6], "accelerator"),
        ([1, 3], [0.4, 0.6], "cpu"),
        ([0, 1], [0.4, 0.6], None),
        ([3, 2], [0.7, 0.3], None),
    ],
)
def test_sequential_hybrid_routing_cases(tmp_path, expert_ids, routing_weights, zero_side):
    coordinator, weights, mapping = _coordinator(tmp_path)
    torch.manual_seed(tiny_fixture.SEED + sum(expert_ids))
    hidden_cpu = torch.randn(1, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    ids_cpu = torch.tensor([expert_ids], dtype=torch.int64)
    route_weights_cpu = torch.tensor([routing_weights], dtype=torch.float32)

    result = coordinator.forward_sequential(
        hidden_cpu.to("npu"),
        ids_cpu.to("npu"),
        route_weights_cpu.to("npu"),
    )
    expected = _reference(hidden_cpu, ids_cpu, route_weights_cpu, weights, mapping)
    _assert_numerical(result.output.cpu(), expected)
    if zero_side == "accelerator":
        assert torch.count_nonzero(result.accelerator_contribution).item() == 0
    elif zero_side == "cpu":
        assert torch.count_nonzero(result.cpu_contribution).item() == 0


def test_sequential_contributions_add_to_output_without_reweighting(tmp_path):
    coordinator, weights, mapping = _coordinator(tmp_path)
    hidden_cpu = torch.randn(4, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16)
    ids_cpu = torch.tensor([[0, 2], [1, 3], [0, 1], [3, 2]], dtype=torch.int64)
    route_weights_cpu = torch.tensor([[0.4, 0.6], [0.4, 0.6], [0.4, 0.6], [0.7, 0.3]])
    result = coordinator.forward_sequential(
        hidden_cpu.to("npu"), ids_cpu.to("npu"), route_weights_cpu.to("npu")
    )

    cpu_reference = _reference(hidden_cpu, ids_cpu, route_weights_cpu, weights, mapping, owned={0, 2})
    npu_reference = _reference(hidden_cpu, ids_cpu, route_weights_cpu, weights, mapping, owned={1, 3})
    _assert_numerical(result.cpu_contribution.cpu(), cpu_reference)
    _assert_numerical(result.accelerator_contribution.cpu(), npu_reference)
    torch.testing.assert_close(
        result.output,
        result.cpu_contribution + result.accelerator_contribution,
        rtol=0,
        atol=0,
    )
