from __future__ import annotations

import gc

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


def test_stream_callback_create_destroy_100_cycles():
    cpuinfer = _cpuinfer()
    states = []
    for _ in range(100):
        stream = torch.npu.Stream()
        state = kt_kernel_ext.testing.CPUInferTestTask()
        with torch.npu.stream(stream):
            cpuinfer.submit_with_device_stream(get_current_device_stream_handle("npu"), state.task(0))
            cpuinfer.sync_with_device_stream(get_current_device_stream_handle("npu"))
        stream.synchronize()
        assert state.completions == 1
        states.append(state)
        del stream
    gc.collect()
    assert sum(state.completions for state in states) == 100


def test_cpuinfer_callback_create_destroy_20_cycles():
    stream = torch.npu.current_stream()
    for _ in range(20):
        cpuinfer = _cpuinfer()
        state = kt_kernel_ext.testing.CPUInferTestTask()
        handle = get_current_device_stream_handle("npu")
        cpuinfer.submit_with_device_stream(handle, state.task(0))
        cpuinfer.sync_with_device_stream(handle)
        stream.synchronize()
        cpuinfer.sync()
        assert state.completions == 1
        del cpuinfer
        gc.collect()


def test_null_stream_launch_error_is_surfaced_without_leaking_sync_state():
    cpuinfer = _cpuinfer()
    with pytest.raises(ValueError, match="non-zero"):
        cpuinfer.sync_with_device_stream(0)


def test_callback_exception_is_rethrown_on_host_thread():
    cpuinfer = _cpuinfer()
    state = kt_kernel_ext.testing.CPUInferTestTask()
    cpuinfer.submit_with_device_stream(
        get_current_device_stream_handle("npu"),
        state.failing_callback_task(),
    )
    torch.npu.current_stream().synchronize()
    with pytest.raises(RuntimeError, match="intentional CPUInfer host callback failure"):
        cpuinfer.rethrow_device_callback_error()
