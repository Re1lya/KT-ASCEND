from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
kt_kernel = pytest.importorskip("kt_kernel")

from kt_kernel import FixedExpertPlacement, KTMoEWrapper, TorchNPUExpertProvider
from kt_kernel.utils.llamafile import LlamafileMoEWrapper
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.layers.moe.kt_ep_wrapper import (
    KTEPWrapperMethod,
    SharedStagingBuffer,
    layer_needs_kt_wrapper,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput


PARENT_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_SCRIPT = (
    PARENT_ROOT
    / "kt-kernel"
    / "test"
    / "fixtures"
    / "tiny_moe"
    / "create_tiny_moe_gguf_fixture.py"
)
SPEC = importlib.util.spec_from_file_location("round3_kt_ep_fixture", FIXTURE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
tiny_fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tiny_fixture)


class _Recorder:
    def on_gpu_expert_mask(self, _layer_idx, _mask):
        pass


def test_kt_wrapper_is_skipped_for_all_accelerator_layers():
    assert not layer_needs_kt_wrapper(torch.ones(64, dtype=torch.bool))
    one_cpu_expert = torch.ones(64, dtype=torch.bool)
    one_cpu_expert[6] = False
    assert layer_needs_kt_wrapper(one_cpu_expert)


class _TorchNPUExpertMethod:
    """Small real-NPU provider for the two accelerator-owned fixture experts."""

    def __init__(self, weights, physical_expert_ids):
        local_weights = {
            name: tensor[physical_expert_ids].contiguous()
            for name, tensor in weights.items()
        }
        placement = FixedExpertPlacement(
            len(physical_expert_ids),
            torch.ones(len(physical_expert_ids), dtype=torch.bool),
        )
        self.provider = TorchNPUExpertProvider(placement, local_weights)

    def apply(self, _layer, dispatch_output):
        hidden_states = dispatch_output.hidden_states
        topk_weights, local_expert_ids, _ = dispatch_output.topk_output
        valid_routes = local_expert_ids >= 0
        output = self.provider.forward(
            hidden_states,
            local_expert_ids.clamp_min(0),
            topk_weights * valid_routes.to(topk_weights.dtype),
        )
        return StandardCombineInput(hidden_states=output)


def _reference(hidden_states, expert_ids, routing_weights, weights):
    rows = []
    for token_index, bf16_input in enumerate(hidden_states.cpu()):
        result = torch.zeros_like(bf16_input.float())
        for route_index in range(expert_ids.shape[1]):
            expert = int(expert_ids[token_index, route_index])
            gate = weights["gate"][expert] @ bf16_input.float()
            up = weights["up"][expert] @ bf16_input.float()
            activated = torch.nn.functional.silu(gate) * up
            result += (
                float(routing_weights[token_index, route_index])
                * (weights["down"][expert] @ activated)
            )
        rows.append(result.to(torch.bfloat16))
    return torch.stack(rows)


def test_kt_ep_wrapper_cpu_npu_routing_cases_run_on_ascend(tmp_path):
    fixture, _ = tiny_fixture.create_fixture(
        tmp_path / "round3-kt-ep.gguf", num_experts=4
    )
    expected_weights = tiny_fixture.generate_weights(num_layers=2, num_experts=4)[0]
    LlamafileMoEWrapper._gguf_loader_instance = None
    accelerator_mask = torch.tensor([False, True, False, True], dtype=torch.bool)
    cpu_wrapper = KTMoEWrapper(
        layer_idx=0,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        moe_intermediate_size=tiny_fixture.DEFAULT_INTERMEDIATE_SIZE,
        gpu_experts_mask=accelerator_mask,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=str(fixture),
        chunked_prefill_size=8,
        max_deferred_experts_per_token=0,
        method="LLAMAFILE",
        numa_nodes=[0],
    )
    cpu_wrapper.load_weights(torch.arange(4, dtype=torch.int32))

    method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
    method.tp_rank = 0
    method.wrapper = cpu_wrapper
    method.kt_expert_lora_enabled = False
    method.num_gpu_experts = 2
    method.gpu_prefill_token_threshold = 0
    method.gpu_experts_mask = accelerator_mask
    method.gpu_experts_mask_cuda = method.gpu_experts_mask.to("npu")
    method.logical_to_gpu_index_cuda = torch.tensor(
        [-1, 0, -1, 1], dtype=torch.int32, device="npu"
    )
    method.gpu_method = _TorchNPUExpertMethod(expected_weights, [1, 3])
    method.kt_config = SimpleNamespace(layer_idx=0)
    method.moe_runner_config = SimpleNamespace(activation="silu")
    device_module = torch.get_device_module(torch.device("npu"))
    method._cpu_stream = device_module.Stream(device=torch.device("npu"))
    method._sync_done_event = device_module.Event()
    method._shared_staging_buffer = SharedStagingBuffer(
        max_tokens=8,
        hidden_size=tiny_fixture.DEFAULT_HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=torch.device("npu"),
    )

    torch.manual_seed(tiny_fixture.SEED + 900)
    hidden_states = torch.randn(
        4,
        tiny_fixture.DEFAULT_HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="npu",
    )
    # CPU-only, NPU-only, mixed, and reversed-mixed rows respectively.
    expert_ids = torch.tensor(
        [[0, 2], [1, 3], [0, 1], [3, 2]], dtype=torch.int64, device="npu"
    )
    routing_weights = torch.tensor(
        [[0.4, 0.6]], dtype=torch.float32, device="npu"
    ).repeat(4, 1)
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=routing_weights,
            topk_ids=expert_ids,
            router_logits=torch.empty(4, 4, dtype=torch.float32, device="npu"),
        ),
    )

    previous_recorder = get_global_expert_distribution_recorder()
    set_global_expert_distribution_recorder(_Recorder())
    try:
        method.apply(SimpleNamespace(), dispatch_output)
        device_module.current_stream().synchronize()
        actual = method.apply(SimpleNamespace(), dispatch_output).hidden_states
        device_module.current_stream().synchronize()
    finally:
        set_global_expert_distribution_recorder(previous_recorder)

    expected = _reference(
        hidden_states, expert_ids.cpu(), routing_weights.cpu(), expected_weights
    )
    row_errors = (actual.cpu().float() - expected.float()).abs()
    print(
        "row_max_abs=",
        [float(value) for value in row_errors.amax(dim=1)],
        "actual_norms=",
        [float(value) for value in actual.cpu().float().norm(dim=1)],
        "expected_norms=",
        [float(value) for value in expected.float().norm(dim=1)],
    )
    torch.testing.assert_close(
        actual.cpu().to(torch.bfloat16), expected, rtol=1e-2, atol=1e-3
    )
    assert torch.isfinite(actual).all()
