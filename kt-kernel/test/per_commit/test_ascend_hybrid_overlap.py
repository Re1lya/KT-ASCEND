# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import (
    FixedExpertPlacement,
    HybridMoECoordinator,
    KTMoEWrapper,
    TorchNPUExpertProvider,
    get_current_device_stream_handle,
    kt_kernel_ext,
)
from kt_kernel.utils.llamafile import LlamafileMoEWrapper


FIXTURE_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_moe" / "create_tiny_moe_gguf_fixture.py"
SPEC = importlib.util.spec_from_file_location("tiny_moe_hybrid_overlap", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tiny_fixture
SPEC.loader.exec_module(tiny_fixture)


def _coordinator(tmp_path, accelerator_mask=None):
    fixture, _ = tiny_fixture.create_fixture(tmp_path / "hybrid-overlap.gguf", num_experts=4)
    weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    mapping = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
    mask = (
        torch.tensor([False, True, False, True], dtype=torch.bool)
        if accelerator_mask is None
        else torch.tensor(accelerator_mask, dtype=torch.bool)
    )
    placement = FixedExpertPlacement(4, mask, mapping)
    LlamafileMoEWrapper._gguf_loader_instance = None
    wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        moe_intermediate_size=tiny_fixture.DEFAULT_INTERMEDIATE_SIZE,
        gpu_experts_mask=mask,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=str(fixture),
        chunked_prefill_size=256,
        max_deferred_experts_per_token=0,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    wrapper.load_weights(mapping)
    provider = TorchNPUExpertProvider(placement, weights)
    return HybridMoECoordinator(wrapper, provider, placement)


def _inputs(qlen):
    torch.manual_seed(tiny_fixture.SEED + qlen + 500)
    hidden = torch.randn(qlen, tiny_fixture.DEFAULT_HIDDEN_SIZE, dtype=torch.bfloat16, device="npu")
    route_matrix = torch.tensor([[0, 2], [1, 3], [0, 1], [3, 2]], dtype=torch.int64, device="npu")
    expert_ids = route_matrix[torch.arange(qlen, device="npu").remainder(4)]
    routing_weights = torch.tensor([[0.4, 0.6]], dtype=torch.float32, device="npu").repeat(qlen, 1)
    return hidden, expert_ids, routing_weights


@pytest.mark.parametrize("qlen", [1, 8, 32])
def test_overlapped_hybrid_matches_sequential(tmp_path, qlen):
    coordinator = _coordinator(tmp_path)
    hidden, expert_ids, routing_weights = _inputs(qlen)
    sequential = coordinator.forward_sequential(hidden, expert_ids, routing_weights)
    overlapped = coordinator.forward_overlapped(
        hidden,
        expert_ids,
        routing_weights,
        get_current_device_stream_handle("npu"),
    )
    torch.npu.current_stream().synchronize()

    torch.testing.assert_close(overlapped.output, sequential.output, rtol=0, atol=0)
    torch.testing.assert_close(overlapped.cpu_contribution, sequential.cpu_contribution, rtol=0, atol=0)
    torch.testing.assert_close(
        overlapped.accelerator_contribution,
        sequential.accelerator_contribution,
        rtol=0,
        atol=0,
    )


def test_real_cpu_and_npu_expert_intervals_overlap(tmp_path):
    coordinator = _coordinator(tmp_path)
    hidden, expert_ids, routing_weights = _inputs(256)
    handle = get_current_device_stream_handle("npu")
    stream = torch.npu.current_stream()

    # Warm up NPU kernels and the CPU grouped-prefill path outside measurement.
    coordinator.forward_overlapped(hidden, expert_ids, routing_weights, handle)
    stream.synchronize()

    cpu_start = kt_kernel_ext.testing.CPUInferTestTask()
    cpu_finish = kt_kernel_ext.testing.CPUInferTestTask()
    npu_start = torch.npu.Event(enable_timing=True)
    npu_finish = torch.npu.Event(enable_timing=True)

    wall_start = time.monotonic_ns()
    coordinator.cpu_wrapper.cpu_infer.submit_with_device_stream(handle, cpu_start.task(0))
    coordinator.cpu_wrapper.submit_forward(hidden, expert_ids, routing_weights, handle)
    coordinator.cpu_wrapper.cpu_infer.submit_with_device_stream(handle, cpu_finish.task(0))
    npu_start.record()
    npu_output = coordinator.accelerator_provider.forward(hidden, expert_ids, routing_weights)
    npu_finish.record()
    cpu_output = coordinator.cpu_wrapper.sync_forward(hidden, handle)
    merged = cpu_output + npu_output
    stream.synchronize()
    wall_finish = time.monotonic_ns()

    assert torch.isfinite(merged).all()
    assert cpu_start.completions == 1 and cpu_finish.completions == 1
    cpu_ms = (cpu_finish.finish_ns - cpu_start.start_ns) / 1_000_000
    npu_ms = npu_start.elapsed_time(npu_finish)
    wall_ms = (wall_finish - wall_start) / 1_000_000
    overlap_lower_bound_ms = cpu_ms + npu_ms - wall_ms
    metrics = {
        "cpu_interval_ms": cpu_ms,
        "npu_interval_ms": npu_ms,
        "wall_ms": wall_ms,
        "overlap_lower_bound_ms": overlap_lower_bound_ms,
    }
    print(json.dumps(metrics, sort_keys=True))
    assert cpu_ms > 0
    assert npu_ms > 0
    assert overlap_lower_bound_ms > 0.1
