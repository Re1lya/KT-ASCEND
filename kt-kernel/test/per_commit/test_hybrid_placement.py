# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from kt_kernel import FixedExpertPlacement


def _placement(mapping=None):
    return FixedExpertPlacement(
        4,
        torch.tensor([False, True, False, True], dtype=torch.bool),
        mapping,
    )


def test_fixed_placement_partitions_every_expert_once():
    placement = _placement()
    assert placement.cpu_experts == (0, 2)
    assert placement.accelerator_experts == (1, 3)
    assert set(placement.cpu_experts).isdisjoint(placement.accelerator_experts)
    assert set(placement.cpu_experts) | set(placement.accelerator_experts) == set(range(4))


def test_global_route_partition_uses_sentinel_and_weight_mask():
    placement = _placement()
    expert_ids = torch.tensor([[0, 1], [3, 2]], dtype=torch.int64)
    weights = torch.tensor([[0.4, 0.6], [0.7, 0.3]], dtype=torch.float32)
    assert placement.cpu_route_ids(expert_ids, weights).tolist() == [[0, -1], [-1, 2]]
    torch.testing.assert_close(
        placement.accelerator_route_weights(expert_ids, weights),
        torch.tensor([[0.0, 0.6], [0.7, 0.0]]),
    )


@pytest.mark.parametrize(
    "mapping",
    [
        [0, 1, 2, 3],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
        [1, 3, 0, 2],
    ],
)
def test_physical_to_logical_permutations(mapping):
    placement = _placement(mapping)
    assert placement.physical_to_logical_map.tolist() == mapping


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (torch.tensor([False, True]), "shape"),
        (torch.tensor([0, 1, 0, 1], dtype=torch.int64), "torch.bool"),
        (torch.tensor([[False, True, False, True]]), "shape"),
    ],
)
def test_invalid_placement_mask_is_rejected(mask, match):
    with pytest.raises(ValueError, match=match):
        FixedExpertPlacement(4, mask)


@pytest.mark.parametrize("mapping", [[0, 1, 1, 3], [0, 1, 2], [0, 1, 2, 4]])
def test_invalid_mapping_is_rejected(mapping):
    with pytest.raises(ValueError, match="physical_to_logical_map"):
        _placement(mapping)


@pytest.mark.parametrize(
    ("ids", "weights", "match"),
    [
        (torch.tensor([[0, 4]]), torch.tensor([[0.5, 0.5]]), "global expert IDs"),
        (torch.tensor([[-1, 1]]), torch.tensor([[0.5, 0.5]]), "global expert IDs"),
        (torch.tensor([[0, 1]]), torch.tensor([[float("nan"), 0.5]]), "finite"),
    ],
)
def test_invalid_routes_are_rejected(ids, weights, match):
    with pytest.raises(ValueError, match=match):
        _placement().validate_routes(ids, weights)
