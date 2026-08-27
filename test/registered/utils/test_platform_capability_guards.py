# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from sglang.srt.utils.common import mxfp8_block_convert_required


def test_mxfp8_capability_probe_does_not_initialize_unavailable_cuda():
    mxfp8_block_convert_required.cache_clear()
    with (
        patch("torch.version.hip", None),
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "torch.cuda.get_device_capability",
            side_effect=AssertionError("CUDA must not be initialized"),
        ),
    ):
        assert mxfp8_block_convert_required() is False
    mxfp8_block_convert_required.cache_clear()
