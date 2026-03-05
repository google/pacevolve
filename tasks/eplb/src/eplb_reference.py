# SPDX-License-Identifier: Apache-2.0
"""
Reference EPLB implementation using tensorized zigzag (snake) pattern.

Matches the paper's "internal reference implementation" description:
- Replaces Python for-loops with PyTorch tensor operations
- Uses zigzag/staggered placement to interleave high-load and low-load experts
- Expected: ~19.6ms runtime, same balancedness as initial (0.66 on paper's trace)
"""

import torch


def balanced_packing_zigzag(
    weight: torch.Tensor, num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Tensorized packing with zigzag pattern (paper Fig 5b).
    Items sorted by weight descending; assigned to packs in snake order.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    device = weight.device
    # Sort by weight descending: indices[layer, pos] = original group id
    indices = weight.float().sort(-1, descending=True).indices

    # Zigzag (paper Fig 5b): pack = idx_in_block (even block) or reverse (odd)
    pos = torch.arange(num_groups, dtype=torch.int64, device=device)
    block_id = pos // num_packs
    idx_in_block = pos % num_packs
    is_even_block = (block_id % 2) == 0
    pack_for_pos = torch.where(
        is_even_block, idx_in_block, num_packs - 1 - idx_in_block
    )
    rank_for_pos = block_id  # one item per pack per block

    # Scatter: pack_index[layer, indices[layer, pos]] = pack_for_pos[pos]
    pack_index = torch.zeros_like(indices, dtype=torch.int64, device=device)
    rank_in_pack = torch.zeros_like(indices, dtype=torch.int64, device=device)
    pack_index.scatter_(
        1, indices, pack_for_pos.unsqueeze(0).expand(num_layers, -1)
    )
    rank_in_pack.scatter_(
        1, indices, rank_for_pos.unsqueeze(0).expand(num_layers, -1)
    )
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same as initial - replicate overloaded experts."""
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(
        n, 1
    )
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """Hierarchical packing with zigzag at all stages."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(
                perm.size(1), dtype=torch.int64, device=perm.device
            ).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes (zigzag)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing_zigzag(
        tokens_per_group, num_nodes
    )
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack)
            * group_size
        ).unsqueeze(-1)
        + torch.arange(
            group_size, dtype=torch.int64, device=group_pack_index.device
        )
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate experts
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs (zigzag)
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing_zigzag(
        tokens_per_phy, num_gpus // num_nodes
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entry point - same signature as eplb_1.py."""
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_replicas, dtype=torch.int64, device=log2phy.device
        ).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
