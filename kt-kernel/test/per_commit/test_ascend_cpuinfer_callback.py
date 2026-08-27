from __future__ import annotations

import json
import time

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import get_current_device_stream_handle, kt_kernel_ext


def _cpuinfer():
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = 1
    config.subpool_numa_map = [0]
    config.subpool_thread_count = [1]
    return kt_kernel_ext.CPUInfer(config)


def test_cpuinfer_submit_and_sync_callbacks():
    cpuinfer = _cpuinfer()
    state = kt_kernel_ext.testing.CPUInferTestTask()
    stream = torch.npu.current_stream()
    handle = get_current_device_stream_handle("npu")

    cpuinfer.submit_with_device_stream(handle, state.task(25))
    cpuinfer.sync_with_device_stream(handle)
    stream.synchronize()

    assert state.completions == 1
    assert state.start_ns > 0
    assert state.finish_ns >= state.start_ns + 20_000_000
    assert get_current_device_stream_handle("npu") == handle


def test_cpuinfer_callback_1000_cycles():
    cpuinfer = _cpuinfer()
    stream = torch.npu.current_stream()
    handle = get_current_device_stream_handle("npu")
    states = []
    for _ in range(1000):
        state = kt_kernel_ext.testing.CPUInferTestTask()
        states.append(state)
        cpuinfer.submit_with_device_stream(handle, state.task(0))
        cpuinfer.sync_with_device_stream(handle)
    stream.synchronize()
    assert sum(state.completions for state in states) == 1000
    assert all(state.finish_ns >= state.start_ns > 0 for state in states)


def _enqueue_matmuls(left: torch.Tensor, right: torch.Tensor, repeats: int):
    result = None
    for _ in range(repeats):
        result = torch.mm(left, right)
    return result


def _event_duration_ms(left: torch.Tensor, right: torch.Tensor, repeats: int) -> float:
    start = torch.npu.Event(enable_timing=True)
    finish = torch.npu.Event(enable_timing=True)
    start.record()
    result = _enqueue_matmuls(left, right, repeats)
    finish.record()
    finish.synchronize()
    assert result is not None
    return start.elapsed_time(finish)


def test_cpu_and_npu_work_intervals_overlap():
    torch.manual_seed(20260827)
    left = torch.randn((4096, 4096), dtype=torch.bfloat16, device="npu")
    right = torch.randn((4096, 4096), dtype=torch.bfloat16, device="npu")
    _event_duration_ms(left, right, 1)  # compile/warm-up outside the measurement

    repeats = 4
    npu_ms = _event_duration_ms(left, right, repeats)
    while npu_ms < 25.0 and repeats < 256:
        repeats *= 2
        npu_ms = _event_duration_ms(left, right, repeats)
    assert npu_ms >= 20.0

    cpuinfer = _cpuinfer()
    state = kt_kernel_ext.testing.CPUInferTestTask()
    handle = get_current_device_stream_handle("npu")
    start_event = torch.npu.Event(enable_timing=True)
    finish_event = torch.npu.Event(enable_timing=True)

    wall_start = time.monotonic_ns()
    cpuinfer.submit_with_device_stream(handle, state.task(150))
    start_event.record()
    result = _enqueue_matmuls(left, right, repeats)
    finish_event.record()
    cpuinfer.sync_with_device_stream(handle)
    torch.npu.current_stream().synchronize()
    wall_finish = time.monotonic_ns()
    assert result is not None

    cpu_ms = (state.finish_ns - state.start_ns) / 1_000_000
    npu_ms = start_event.elapsed_time(finish_event)
    wall_ms = (wall_finish - wall_start) / 1_000_000
    overlap_lower_bound_ms = cpu_ms + npu_ms - wall_ms
    metrics = {
        "cpu_ms": cpu_ms,
        "npu_ms": npu_ms,
        "wall_ms": wall_ms,
        "overlap_lower_bound_ms": overlap_lower_bound_ms,
        "matmul_repeats": repeats,
    }
    print(json.dumps(metrics, sort_keys=True))
    assert state.completions == 1
    assert cpu_ms >= 140.0
    assert npu_ms >= 20.0
    assert overlap_lower_bound_ms >= 5.0
