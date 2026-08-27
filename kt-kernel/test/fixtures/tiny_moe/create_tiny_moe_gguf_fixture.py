#!/usr/bin/env python3
"""Create a deterministic, local-only F32 GGUF MoE fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from gguf import GGUFWriter

SEED = 20260827
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_INTERMEDIATE_SIZE = 256


def generate_weights(
    *,
    num_layers: int,
    num_experts: int,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Generate the canonical fixture tensors without reading a GGUF file."""
    torch.manual_seed(SEED)
    weights: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer_idx in range(num_layers):
        weights[layer_idx] = {
            "gate": (torch.randn(num_experts, intermediate_size, hidden_size) * 0.02).float().contiguous(),
            "up": (torch.randn(num_experts, intermediate_size, hidden_size) * 0.02).float().contiguous(),
            "down": (torch.randn(num_experts, hidden_size, intermediate_size) * 0.02).float().contiguous(),
        }
    return weights


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def create_fixture(
    output_path: Path,
    *,
    num_layers: int = 2,
    num_experts: int = 8,
    top_k: int = 2,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
) -> Tuple[Path, dict]:
    """Write a deterministic GGUF fixture and a checksum manifest."""
    if intermediate_size % 256 != 0:
        raise ValueError("intermediate_size must be divisible by LLAMAFILE QK_K=256")
    if not 0 < top_k <= num_experts:
        raise ValueError("top_k must be between 1 and num_experts")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights = generate_weights(
        num_layers=num_layers,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    writer = GGUFWriter(str(output_path), "llama")
    writer.add_block_count(num_layers)
    writer.add_uint32("llama.expert_count", num_experts)
    writer.add_uint32("llama.expert_used_count", top_k)
    writer.add_uint32("llama.embedding_length", hidden_size)
    writer.add_uint32("llama.expert_feed_forward_length", intermediate_size)

    tensor_manifest = {}
    for layer_idx, layer_weights in weights.items():
        tensor_names = {
            "gate": f"blk.{layer_idx}.ffn_gate_exps.weight",
            "up": f"blk.{layer_idx}.ffn_up_exps.weight",
            "down": f"blk.{layer_idx}.ffn_down_exps.weight",
        }
        for projection, tensor_name in tensor_names.items():
            tensor = layer_weights[projection]
            array = np.asarray(tensor.numpy(), dtype=np.float32, order="C")
            writer.add_tensor(tensor_name, array)
            tensor_manifest[tensor_name] = {
                "shape": list(tensor.shape),
                "dtype": "float32",
                "ggml_type": "F32",
                "sha256": _tensor_sha256(tensor),
            }

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    manifest = {
        "seed": SEED,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "quant_type": "F32",
        "tensors": tensor_manifest,
        "gguf_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    output, manifest = create_fixture(
        args.output,
        num_layers=args.layers,
        num_experts=args.experts,
        top_k=args.top_k,
    )
    print(json.dumps({"output": str(output), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
