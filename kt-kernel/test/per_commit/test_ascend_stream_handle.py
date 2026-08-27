from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_npu_stream_handle_uses_public_borrowed_integer(monkeypatch):
    import kt_kernel.device_stream as device_stream

    stream = SimpleNamespace(npu_stream=0x123456789ABC)
    monkeypatch.setattr(device_stream.torch, "npu", SimpleNamespace(current_stream=lambda: stream), raising=False)
    assert device_stream.get_current_device_stream_handle("npu") == stream.npu_stream
    assert stream.npu_stream == 0x123456789ABC


def test_stream_handle_rejects_unsupported_device():
    from kt_kernel.device_stream import get_current_device_stream_handle

    with pytest.raises(ValueError, match="unsupported device stream type"):
        get_current_device_stream_handle("cpu")


def test_stream_handle_rejects_null_public_handle(monkeypatch):
    import kt_kernel.device_stream as device_stream

    monkeypatch.setattr(
        device_stream.torch,
        "npu",
        SimpleNamespace(current_stream=lambda: SimpleNamespace(npu_stream=0)),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="non-zero public npu_stream"):
        device_stream.get_current_device_stream_handle("npu")


@pytest.mark.skipif("torch_npu" not in sys.modules, reason="requires an explicitly imported torch_npu runtime")
def test_real_ascend_stream_handle_is_stable_and_non_owning():
    import torch

    from kt_kernel.device_stream import get_current_device_stream_handle

    stream = torch.npu.current_stream()
    handle = get_current_device_stream_handle("npu")
    assert handle == stream.npu_stream
    assert 0 < handle < (1 << 64)

    value = torch.arange(16, dtype=torch.float32, device="npu") + 1
    stream.synchronize()
    assert get_current_device_stream_handle("npu") == handle
    assert torch.equal(value.cpu(), torch.arange(1, 17, dtype=torch.float32))
