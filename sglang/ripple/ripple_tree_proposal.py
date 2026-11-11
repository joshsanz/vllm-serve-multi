"""Experimental batched DFlash selector lattice to native Eagle tree topology.

No model, scheduler, allocator, attention, or acceptance code lives here. The
builder uses fixed-shape Torch device operations and can run during CUDA graph
capture. It is an executable reference prototype, not a measured fast builder.

Selection follows historical v23's deterministic best-first policy: retain the
top C selector children, rank paths by the product of their draft probabilities,
keep N-1 nodes globally, then lay out parents before children. Native Eagle's
target-only tree verifier consumes the topology; proposal probabilities do not
replace target probabilities. Ties use lower candidate index, earlier depth,
earlier parent beam, then earlier child rank.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TreeProposal:
    tokens: torch.Tensor                  # [B,N], int64, node zero is anchor
    parents: torch.Tensor                 # [B,N], local node indices
    depths: torch.Tensor                  # [B,N], anchor depth zero
    valid: torch.Tensor                   # [B,N], unused nodes cannot be reached
    scores: torch.Tensor                  # [B,N], float64 path products
    candidate_indices: torch.Tensor       # [B,N], selector slot index
    ranks: torch.Tensor                   # [B,N], sibling score rank
    ancestors: torch.Tensor               # [B,N,N], self + ancestors
    retrieve_index: torch.Tensor          # [B,N], flattened B*N target row index
    retrieve_next_token: torch.Tensor     # [B,N], first child local index or -1
    retrieve_next_sibling: torch.Tensor   # [B,N], next sibling local index or -1


def build_tree(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    anchors: torch.Tensor,
    *,
    node_budget: int = 64,
    children: int = 16,
    draft_temperature: float = 1.5,
    max_depth: Optional[torch.Tensor] = None,
    max_batch_size: int = 4,
) -> TreeProposal:
    """Build B independent irregular trees without data-dependent host reads.

    candidate_ids is [B,D,K], scores is [B,D,K,K] (parent index, child index).
    At depth one all parent rows represent the same anchor; row zero is used.
    Optional max_depth is [B] on the same device, enabling fixed-shape near-EOS
    restriction. Inputs must have unique token IDs within each candidate slot.
    """
    if candidate_ids.ndim != 3 or scores.ndim != 4:
        raise ValueError("Expected candidate_ids[B,D,K] and scores[B,D,K,K]")
    b, d, k = candidate_ids.shape
    if tuple(scores.shape) != (b, d, k, k) or tuple(anchors.shape) != (b,):
        raise ValueError("Selector tensor shapes do not match")
    if type(max_batch_size) is not int or max_batch_size not in (4, 8):
        raise ValueError("max_batch_size must be 4 or the explicit experimental limit 8")
    if not (1 <= b <= max_batch_size and d >= 1 and node_budget >= 2 and 1 <= children <= k):
        raise ValueError(f"Expected B1–{max_batch_size}, D>=1, N>=2, and 1<=children<=K")
    if not draft_temperature > 0 or not float(draft_temperature) < float("inf"):
        raise ValueError("draft_temperature must be finite and positive")
    if max_depth is not None and tuple(max_depth.shape) != (b,):
        raise ValueError("max_depth must have shape [B]")
    device = scores.device
    m = node_budget - 1
    bi = torch.arange(b, device=device)[:, None]
    beam = torch.arange(m, device=device)[None, :]
    # Historical ranking computes softmax in float32 and products in float64.
    probs = torch.softmax(scores.float() / float(draft_temperature), dim=-1)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :children]
    ordered_probs = probs.gather(-1, order).double()
    prev_probs = torch.where(beam == 0, 1.0, -1.0).expand(b, -1).double()
    prev_ci = torch.zeros((b, m), dtype=torch.long, device=device)
    level_values, level_ci, level_parent, level_ranks = [], [], [], []
    for depth in range(d):
        q = ordered_probs[:, depth][bi, prev_ci]
        child_ci = order[:, depth][bi, prev_ci]
        live = prev_probs[..., None] >= 0
        if max_depth is not None:
            live = live & (depth < max_depth[:, None, None])
        values = torch.where(live, prev_probs[..., None] * q, -1.0)
        flat = values.reshape(b, m * children)
        selected = torch.argsort(flat, dim=-1, descending=True, stable=True)[:, :m]
        prev_probs = flat.gather(1, selected)
        parent = selected // children
        rank = selected % children
        prev_ci = child_ci.reshape(b, -1).gather(1, selected)
        level_values.append(prev_probs)
        level_ci.append(prev_ci)
        level_parent.append(parent)
        level_ranks.append(rank)
    all_values = torch.cat(level_values, dim=1)
    all_ci = torch.cat(level_ci, dim=1)
    all_parent = torch.cat(level_parent, dim=1)
    all_ranks = torch.cat(level_ranks, dim=1)
    # Stable global ordering makes ancestors win exact product ties.
    chosen = torch.argsort(all_values, dim=-1, descending=True, stable=True)[:, :m]
    chosen = torch.sort(chosen, dim=-1).values  # depth then parent-before-child
    selected_values = all_values.gather(1, chosen)
    selected_valid = selected_values >= 0
    selected_depth = chosen // m + 1
    selected_ci = all_ci.gather(1, chosen)
    selected_rank = all_ranks.gather(1, chosen)
    previous_beam = (selected_depth - 2) * m + all_parent.gather(1, chosen)
    mapping = torch.zeros((b, d * m), dtype=torch.long, device=device)
    mapping.scatter_(1, chosen, torch.where(selected_valid, beam + 1, 0))
    parents = mapping.gather(1, previous_beam.clamp_min(0))
    parents = torch.where((selected_depth == 1) | ~selected_valid, 0, parents)
    selected_tokens = candidate_ids[bi, selected_depth - 1, selected_ci]
    selected_tokens = torch.where(selected_valid, selected_tokens, anchors[:, None])
    zeros = torch.zeros((b, 1), dtype=torch.long, device=device)
    tokens = torch.cat((anchors[:, None].long(), selected_tokens.long()), dim=1)
    parents = torch.cat((zeros, parents), dim=1)
    depths = torch.cat((zeros, torch.where(selected_valid, selected_depth, 1)), dim=1)
    valid = torch.cat((torch.ones((b, 1), dtype=torch.bool, device=device), selected_valid), dim=1)
    path_scores = torch.cat((torch.ones((b, 1), dtype=torch.float64, device=device), selected_values), dim=1)
    candidate_indices = torch.cat((zeros, selected_ci), dim=1)
    ranks = torch.cat((zeros, selected_rank), dim=1)
    nodes = torch.arange(node_budget, device=device)
    ancestors = torch.eye(node_budget, dtype=torch.bool, device=device)[None].expand(b, -1, -1).clone()
    path = nodes[None].expand(b, -1)
    for _ in range(d):
        path = parents.gather(1, path)
        ancestors.scatter_(2, path[..., None], True)
    # Child and sibling order is explicit selector rank, independent of layout.
    child_ids = nodes[None, None, :].expand(b, node_budget, -1)
    eligible = valid[:, None, :] & (child_ids != 0)
    key = ranks[:, None, :] * node_budget + child_ids
    sentinel = node_budget * (k + 1)
    child = eligible & (parents[:, None, :] == nodes[None, :, None])
    first_key = torch.where(child, key, sentinel).amin(dim=-1)
    first_child = torch.where(first_key < sentinel, first_key % node_budget, -1)
    sibling = (eligible & (parents[:, None, :] == parents[:, :, None])
               & (ranks[:, None, :] > ranks[:, :, None]) & valid[:, :, None])
    sibling_key = torch.where(sibling, key, sentinel).amin(dim=-1)
    next_sibling = torch.where(sibling_key < sentinel, sibling_key % node_budget, -1)
    next_sibling[:, 0] = -1
    retrieve_index = nodes[None, :].expand(b, -1) + torch.arange(b, device=device)[:, None] * node_budget
    return TreeProposal(tokens, parents, depths, valid, path_scores, candidate_indices,
                        ranks, ancestors, retrieve_index, first_child, next_sibling)
