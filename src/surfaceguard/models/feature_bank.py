from typing import Optional, Tuple
import numpy as np
import torch
from surfaceguard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def subsample_memory_bank(
    features: torch.Tensor,
    sampling_ratio: float = 0.10,
    method: str = "coreset",
    seed: int = 42,
) -> torch.Tensor:
    """Sub-sample the memory bank to reduce memory and accelerate nearest-neighbor retrieval."""
    N, D = features.shape
    target_count = max(10, int(N * sampling_ratio))

    if target_count >= N:
        return features

    torch.manual_seed(seed)

    if method == "random":
        perm = torch.randperm(N)[:target_count]
        return features[perm]

    # Fast Approximate Coreset Selection
    # For CPU efficiency, cap target vectors to maximum 1500 prototypes per category
    effective_target = min(target_count, 1500)
    logger.info(f"Subsampling memory bank from {N} to {effective_target} vectors via {method}...")

    # Pool candidate points for coreset
    candidate_pool_size = min(N, effective_target * 5)
    candidate_indices = torch.randperm(N)[:candidate_pool_size]
    candidates = features[candidate_indices]

    selected_indices = [torch.randint(0, candidate_pool_size, (1,)).item()]
    min_distances = torch.full((candidate_pool_size,), float("inf"), device=candidates.device)

    # Iterative k-center selection
    for _ in range(effective_target - 1):
        last_selected = candidates[selected_indices[-1]].unsqueeze(0)
        dist_to_last = torch.norm(candidates - last_selected, dim=1)
        min_distances = torch.minimum(min_distances, dist_to_last)
        next_selected = torch.argmax(min_distances).item()
        selected_indices.append(next_selected)

    coreset = candidates[selected_indices]
    logger.info(f"Memory bank successfully reduced to shape {coreset.shape}")
    return coreset


def compute_min_distances_chunked(
    query_patches: torch.Tensor,
    memory_bank: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Compute minimum Euclidean distance for each query patch against memory bank in safe chunks.
    
    query_patches: (Q, D)
    memory_bank: (M, D)
    Returns: min_distances of shape (Q,)
    """
    Q, D = query_patches.shape
    M = memory_bank.shape[0]
    min_dists = torch.empty(Q, device=query_patches.device, dtype=torch.float32)

    # Precompute memory bank squared norms: (M,)
    bank_sq_norms = torch.sum(memory_bank ** 2, dim=1)

    for start_idx in range(0, Q, chunk_size):
        end_idx = min(start_idx + chunk_size, Q)
        q_chunk = query_patches[start_idx:end_idx]  # (B_chunk, D)

        # ||q - m||^2 = ||q||^2 - 2 * q @ m.T + ||m||^2
        q_sq_norms = torch.sum(q_chunk ** 2, dim=1, keepdim=True)  # (B_chunk, 1)
        cross_term = torch.matmul(q_chunk, memory_bank.t())  # (B_chunk, M)

        dists_sq = q_sq_norms - 2.0 * cross_term + bank_sq_norms.unsqueeze(0)
        dists_sq = torch.clamp(dists_sq, min=0.0)

        chunk_min_dist, _ = torch.min(torch.sqrt(dists_sq), dim=1)
        min_dists[start_idx:end_idx] = chunk_min_dist

    return min_dists
