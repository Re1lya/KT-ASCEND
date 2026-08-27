from __future__ import annotations

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import get_current_device_stream_handle, kt_kernel_ext, require_pinned_host_tensor


def _cpuinfer():
    config = kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = 1
    config.subpool_numa_map = [0]
    config.subpool_thread_count = [1]
    return kt_kernel_ext.CPUInfer(config)


def test_pytorch_ascend_pinned_allocator_contract():
    pageable = torch.empty(64, dtype=torch.bfloat16, device="cpu")
    pinned = torch.empty(64, dtype=torch.bfloat16, device="cpu", pin_memory=True)
    assert not pageable.is_pinned()
    assert pinned.is_pinned()
    assert require_pinned_host_tensor(pinned, "pinned") is pinned
    with pytest.raises(ValueError, match="pinned host memory"):
        require_pinned_host_tensor(pageable, "pageable")
    with pytest.raises(ValueError, match="must be a CPU tensor"):
        require_pinned_host_tensor(torch.empty(1, device="npu"), "device")


@pytest.mark.parametrize(
    ("dtype", "source"),
    [
        (torch.bfloat16, lambda: torch.arange(4096, dtype=torch.float32).to(torch.bfloat16)),
        (torch.int64, lambda: torch.arange(4096, dtype=torch.int64).remainder(8)),
        (torch.float32, lambda: torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32)),
    ],
)
def test_async_d2h_patterns_are_visible_before_host_callback(dtype, source):
    expected = source()
    device = expected.to(device="npu", dtype=dtype)
    host = require_pinned_host_tensor(torch.empty_like(expected, device="cpu", pin_memory=True), "d2h_host")
    cpuinfer = _cpuinfer()
    marker = kt_kernel_ext.testing.CPUInferTestTask()
    stream = torch.npu.current_stream()

    host.copy_(device, non_blocking=True)
    cpuinfer.submit_with_device_stream(get_current_device_stream_handle("npu"), marker.task(0))
    cpuinfer.sync_with_device_stream(get_current_device_stream_handle("npu"))
    stream.synchronize()

    assert marker.completions == 1
    assert torch.equal(host, expected)


def test_async_h2d_precedes_subsequent_npu_verification():
    torch.manual_seed(20260827)
    host = require_pinned_host_tensor(
        torch.randn(4096, dtype=torch.float32).to(torch.bfloat16).pin_memory(),
        "h2d_host",
    )
    device = torch.empty_like(host, device="npu")
    device.copy_(host, non_blocking=True)
    verified = device.to(torch.float32) * 2.0 + 1.0
    torch.npu.current_stream().synchronize()
    expected = host.to(torch.float32) * 2.0 + 1.0
    torch.testing.assert_close(verified.cpu(), expected, rtol=0, atol=0)
