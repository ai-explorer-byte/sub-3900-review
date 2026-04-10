from typing import Dict, List, Set
import torch
import torch.nn.functional as F

from src.data_loader import EvidenceUnit


def _get_similarity_matrix(units: List[EvidenceUnit]) -> torch.Tensor:
    embeddings = torch.stack([u.embedding for u in units])
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    return embeddings_norm @ embeddings_norm.T


def mmr_gain(
    candidate_idx: int,
    selected_indices: Set[int],
    units: List[EvidenceUnit],
    sim_matrix: torch.Tensor,
    config: Dict,
) -> float:
    """
    MMR: lambda * imp(u) - (1 - lambda) * max_{v in S} sim(u, v)
    """
    lam = config["lambda_mmr"]
    imp_u = units[candidate_idx].importance

    if not selected_indices:
        max_sim = 0.0
    else:
        sims = [sim_matrix[candidate_idx, j].item() for j in selected_indices]
        max_sim = max(sims)

    return lam * imp_u - (1 - lam) * max_sim


def graphcut_gain(
    candidate_idx: int,
    selected_indices: Set[int],
    units: List[EvidenceUnit],
    sim_matrix: torch.Tensor,
    config: Dict,
) -> float:
    """
    GraphCut: imp(u) - sum_{v in S} sim(u, v)
    Penalizes adding items similar to already selected ones.
    """
    imp_u = units[candidate_idx].importance

    if not selected_indices:
        penalty = 0.0
    else:
        penalty = sum(sim_matrix[candidate_idx, j].item() for j in selected_indices)

    return imp_u - penalty


def facility_location_gain(
    candidate_idx: int,
    selected_indices: Set[int],
    units: List[EvidenceUnit],
    sim_matrix: torch.Tensor,
    config: Dict,
) -> float:
    """
    Facility Location: sum over all units of the improvement in max coverage.
    f(S) = sum_i max_{j in S} sim(i, j)
    Marginal gain = sum_i max(0, sim(i, u) - max_{j in S} sim(i, j))
    """
    n = len(units)
    gain = 0.0

    for i in range(n):
        sim_to_candidate = sim_matrix[i, candidate_idx].item()
        if not selected_indices:
            current_max = 0.0
        else:
            current_max = max(sim_matrix[i, j].item() for j in selected_indices)
        gain += max(0.0, sim_to_candidate - current_max)

    return gain


def dpp_log_det_gain(
    candidate_idx: int,
    selected_indices: Set[int],
    units: List[EvidenceUnit],
    L_matrix: torch.Tensor,
    config: Dict,
) -> float:
    """
    DPP: log det(L_{S+u}) - log det(L_S)
    Uses Schur complement for efficient computation.
    """
    if not selected_indices:
        return torch.log(L_matrix[candidate_idx, candidate_idx] + 1e-10).item()

    S_list = sorted(selected_indices)
    L_S = L_matrix[S_list][:, S_list]

    # Schur complement: det(L_{S+u}) / det(L_S) = L_uu - L_uS @ inv(L_S) @ L_Su
    L_uu = L_matrix[candidate_idx, candidate_idx]
    L_uS = L_matrix[candidate_idx, S_list].unsqueeze(0)  # (1, |S|)
    L_Su = L_matrix[S_list, candidate_idx].unsqueeze(1)  # (|S|, 1)

    # Add small regularization for numerical stability
    L_S_reg = L_S + 1e-6 * torch.eye(len(S_list))
    L_S_inv = torch.linalg.inv(L_S_reg)

    schur = L_uu - (L_uS @ L_S_inv @ L_Su).squeeze()
    schur = max(schur.item(), 1e-10)

    return torch.log(torch.tensor(schur)).item()
