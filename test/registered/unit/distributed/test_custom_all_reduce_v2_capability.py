from unittest.mock import Mock

import pytest
import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce_v2
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _patch_group(monkeypatch, *, world_size, same_node):
    group = object()
    device = object()
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "get_world_size",
        lambda group: world_size,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_supported_world_sizes",
        lambda: (world_size,),
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "in_the_same_node_as",
        lambda group, source_rank: [same_node] * world_size,
    )
    return group, device


@pytest.mark.parametrize(
    ("same_node", "has_fabric_clique", "expected"),
    [
        (False, True, True),
        (False, False, False),
        (True, None, True),
    ],
)
def test_topology_capability(monkeypatch, same_node, has_fabric_clique, expected):
    world_size = 8 if same_node else 16
    group, device = _patch_group(
        monkeypatch,
        world_size=world_size,
        same_node=same_node,
    )

    def is_one_clique(group, device):
        if same_node:
            pytest.fail("intra-node groups do not need a fabric clique")
        return has_fabric_clique

    intra_node_capability = Mock(return_value=True)

    monkeypatch.setattr(
        custom_all_reduce_v2,
        "is_one_nvlink_clique",
        is_one_clique,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        intra_node_capability,
    )

    assert custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
    if same_node:
        intra_node_capability.assert_called_once_with(
            group=group,
            device=device,
            supported_world_size=[world_size],
            cls_name="CustomAllReduceV2",
        )
    else:
        intra_node_capability.assert_not_called()


@pytest.mark.parametrize(
    ("disabled", "has_multicast", "nbytes", "num_blocks", "expected"),
    [
        (False, True, 1024, 4, True),
        (False, True, 1025, 4, False),
        (False, True, 1024, 5, False),
        (False, True, 0, 1, False),
        (False, False, 1024, 4, True),
        (True, None, 1024, 4, False),
    ],
)
def test_supports_push(disabled, has_multicast, nbytes, num_blocks, expected):
    """Push is offered within the plane's inclusive capacity, with or without multicast."""
    comm = object.__new__(custom_all_reduce_v2.CustomAllReduceV2)
    comm.disabled = disabled
    # A disabled communicator never builds its planes, so these stay unset.
    if not disabled:
        comm.has_multicast = has_multicast
        comm.max_push_size = 1024
        comm._push_counter = torch.zeros(4)
    assert comm.supports_push(nbytes=nbytes, num_blocks=num_blocks) is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_pcie_config_selects_only_one_shot_push():
    """Over PCIe P2P the pull kernels never finish and two-shot push is untested,
    so the opt-in config must route every size to one-shot push or to NCCL."""
    from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
        MB,
        AllReduceConfig,
        Heuristic,
    )

    # The table SM70 (no tuned entry) uses: 1 MB one-shot crossovers.
    default = AllReduceConfig(
        graph=Heuristic(1 * MB, 1 * MB, 16 * MB),
        eager=Heuristic(1 * MB, 1 * MB, 16 * MB),
        num_push_blocks=80,
        num_pull_blocks=80,
        num_mc_blocks=None,
    )
    cap = 128 * 1024
    config = custom_all_reduce_v2.pcie_one_shot_push_config(default, cap).clip(
        max_push_bytes=cap, max_pull_bytes=0, world_size=4
    )
    for heuristic in (config.graph, config.eager):
        assert heuristic.one_shot_push_threshold == cap
        assert heuristic.two_shot_push_threshold == 0
        assert heuristic.one_shot_pull_threshold == 0
        assert heuristic.two_shot_pull_threshold == 0
        assert heuristic.mc.max_bytes == 0


@pytest.mark.parametrize(
    ("full_nvlink", "allow_pcie", "expected"),
    [
        (False, False, False),
        (False, True, True),
        # A failed P2P test or an unsupported group stays off whatever is set.
        (None, True, False),
        (True, False, True),
    ],
)
def test_pcie_opt_in_enables_the_communicator(
    monkeypatch, full_nvlink, allow_pcie, expected
):
    """SGLANG_CUSTOM_AR_ALLOW_PCIE must turn the communicator on for GPUs that
    pass the P2P test without NVLink. It used to log that it would and then
    stay on NCCL, with no warning."""
    group, device = _patch_group(monkeypatch, world_size=4, same_node=True)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "can_use_custom_all_reduce_with_nvlink",
        lambda **kwargs: full_nvlink,
    )
    with envs.SGLANG_CUSTOM_AR_ALLOW_PCIE.override(allow_pcie):
        assert (
            custom_all_reduce_v2.can_use_custom_all_reduce_v2(group, device) is expected
        )
