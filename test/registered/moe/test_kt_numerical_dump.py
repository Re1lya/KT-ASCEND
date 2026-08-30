"""Regression tests for opt-in KTransformers numerical instrumentation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod


class TestKTNumericalDump(unittest.TestCase):
    def test_output_stages_preserve_premerge_gpu_contribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "layer17-pass00000.pt"
            torch.save({"schema_version": 1}, output_path)
            method = KTEPWrapperMethod.__new__(KTEPWrapperMethod)
            method._pending_numerical_dump_paths = [output_path]
            cpu_output = torch.tensor([[1.0, 2.0]])
            gpu_output = torch.tensor([[3.0, 4.0]])
            merged_output = cpu_output + gpu_output
            gpu_routes = torch.tensor([[5, 6]])

            method._maybe_dump_numerical_outputs(
                merged_output, cpu_output, gpu_output, gpu_routes
            )

            payload = torch.load(output_path, map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(payload["cpu_output"], cpu_output))
            self.assertTrue(torch.equal(payload["gpu_output"], gpu_output))
            self.assertTrue(torch.equal(payload["merged_output"], merged_output))
            self.assertTrue(torch.equal(payload["gpu_routes"], gpu_routes))
            self.assertEqual(method._pending_numerical_dump_paths, [])


if __name__ == "__main__":
    unittest.main()
