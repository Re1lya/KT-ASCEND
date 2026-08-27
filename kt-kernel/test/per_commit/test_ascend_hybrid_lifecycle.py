# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")

from kt_kernel import get_current_device_stream_handle


OVERLAP_TEST = Path(__file__).with_name("test_ascend_hybrid_overlap.py")
SPEC = importlib.util.spec_from_file_location("hybrid_overlap_lifecycle_helpers", OVERLAP_TEST)
assert SPEC is not None and SPEC.loader is not None
overlap_helpers = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = overlap_helpers
SPEC.loader.exec_module(overlap_helpers)


def _rss_bytes():
    resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def test_overlapped_hybrid_1000_mixed_cycles_and_rss(tmp_path):
    coordinator = overlap_helpers._coordinator(tmp_path)
    hidden, expert_ids, routing_weights = overlap_helpers._inputs(4)
    handle = get_current_device_stream_handle("npu")
    stream = torch.npu.current_stream()

    baseline = None
    for _ in range(20):
        baseline = coordinator.forward_overlapped(hidden, expert_ids, routing_weights, handle).output
    stream.synchronize()
    gc.collect()
    rss_before = _rss_bytes()

    actual = None
    for _ in range(1000):
        actual = coordinator.forward_overlapped(hidden, expert_ids, routing_weights, handle).output
    stream.synchronize()
    gc.collect()
    rss_after = _rss_bytes()

    assert baseline is not None and actual is not None
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert rss_after - rss_before <= 16 * 1024 * 1024
    print({"cycles": 1000, "rss_before": rss_before, "rss_after": rss_after, "rss_delta": rss_after - rss_before})


def test_hybrid_wrapper_recreate_20_alternating_placements(tmp_path):
    masks = ([False, True, False, True], [True, False, True, False])
    for iteration in range(20):
        coordinator = overlap_helpers._coordinator(tmp_path / f"wrapper-{iteration}", masks[iteration % 2])
        hidden, expert_ids, routing_weights = overlap_helpers._inputs(4)
        result = coordinator.forward_overlapped(
            hidden,
            expert_ids,
            routing_weights,
            get_current_device_stream_handle("npu"),
        )
        torch.npu.current_stream().synchronize()
        assert torch.isfinite(result.output).all()
        assert coordinator.placement.accelerator_mask.tolist() == list(masks[iteration % 2])
        del result, coordinator
        gc.collect()


def test_hybrid_stream_create_destroy_100_cycles(tmp_path):
    coordinator = overlap_helpers._coordinator(tmp_path)
    for _ in range(100):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            hidden, expert_ids, routing_weights = overlap_helpers._inputs(4)
            result = coordinator.forward_overlapped(
                hidden,
                expert_ids,
                routing_weights,
                get_current_device_stream_handle("npu"),
            )
        stream.synchronize()
        assert torch.isfinite(result.output).all()
        del result, stream
    gc.collect()


def test_hybrid_rejects_zero_stream_handle(tmp_path):
    coordinator = overlap_helpers._coordinator(tmp_path)
    hidden, expert_ids, routing_weights = overlap_helpers._inputs(1)
    with pytest.raises(ValueError, match="non-zero"):
        coordinator.forward_overlapped(hidden, expert_ids, routing_weights, 0)
