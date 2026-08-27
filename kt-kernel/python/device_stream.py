"""Public framework stream handles for CPUInfer device-stream callbacks."""

from __future__ import annotations

import torch


def get_current_device_stream_handle(device_type: str) -> int:
    """Return a borrowed native stream handle for a supported torch device."""
    normalized = device_type.lower()
    if normalized == "npu":
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("torch.npu is unavailable; import a compatible public torch_npu package first")
        stream = npu.current_stream()
        attribute = "npu_stream"
    elif normalized == "cuda":
        stream = torch.cuda.current_stream()
        attribute = "cuda_stream"
    else:
        raise ValueError(f"unsupported device stream type: {device_type!r}")

    handle = getattr(stream, attribute, None)
    if not isinstance(handle, int) or handle == 0:
        raise RuntimeError(f"current {normalized} stream has no non-zero public {attribute} handle")
    return handle
