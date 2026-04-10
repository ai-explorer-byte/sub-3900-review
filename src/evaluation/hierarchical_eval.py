import logging
from typing import Dict, List

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class AggregateMetricsCounter:
    """Dynamic counter for tracking aggregate evaluation metrics across papers."""

    def __init__(self):
        self._counts: Dict[str, int] = {}
        self._sums: Dict[str, float] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            self._counts[key] = self._counts.get(key, 0) + 1
            self._sums[key] = self._sums.get(key, 0.0) + value

    def get_averages(self) -> Dict[str, float]:
        return {
            key: self._sums[key] / self._counts[key]
            for key in self._counts
        }

    def get_counts(self) -> Dict[str, int]:
        return dict(self._counts)

    def log_summary(self) -> None:
        averages = self.get_averages()
        counts = self.get_counts()
        log.info("Aggregate Evaluation Summary:")
        for key in sorted(averages.keys()):
            log.info(f"  {key}: {averages[key]:.4f} (n={counts[key]})")


def _compute_cosine_similarity(
    queries: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity matrix between queries and keys.

    :param queries: (Q, D) tensor of query embeddings
    :param keys: (K, D) tensor of key embeddings
    :return: (Q, K) tensor of cosine similarities
    """
    queries_norm = F.normalize(queries, p=2, dim=1)
    keys_norm = F.normalize(keys, p=2, dim=1)
    return torch.matmul(queries_norm, keys_norm.T)


def _compute_set_redundancy(selected_embeddings: torch.Tensor) -> float:
    """
    Compute set redundancy as mean of off-diagonal cosine similarities.

    Lower is better - high score (>0.8) means nearly identical chunks.

    :param selected_embeddings: (S, D) tensor of selected embeddings
    :return: Mean of off-diagonal cosine similarities, or 0.0 if fewer than 2 items
    """
    if selected_embeddings.shape[0] < 2:
        return 0.0

    # Compute cosine similarity matrix
    sim_matrix = _compute_cosine_similarity(selected_embeddings, selected_embeddings)

    # Get off-diagonal elements (exclude self-similarity on diagonal)
    n = sim_matrix.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
    off_diagonal = sim_matrix[mask]

    return off_diagonal.mean().item()


def _compute_information_density(selected_phi: torch.Tensor) -> float:
    """
    Compute information density as mean of phi (importance) values.

    Higher is better - dense reasoning (phi~1.0) vs fluff (phi~0.0).

    :param selected_phi: (S,) tensor of phi/importance values for selected items
    :return: Mean phi value, or 0.0 if empty
    """
    if selected_phi.numel() == 0:
        return 0.0
    return selected_phi.mean().item()


def _compute_semantic_volume(selected_embeddings: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute semantic volume (diversity) via log-determinant of similarity kernel.

    Higher is better - measures the "volume" of semantic space covered.

    :param selected_embeddings: (S, D) tensor of selected embeddings
    :param epsilon: Small noise for numerical stability
    :return: Log-determinant of kernel matrix, or 0.0 if fewer than 1 item
    """
    if selected_embeddings.shape[0] < 1:
        return 0.0

    # Compute cosine similarity kernel
    K = _compute_cosine_similarity(selected_embeddings, selected_embeddings)

    # Add small noise to diagonal for numerical stability
    n = K.shape[0]
    K = K + epsilon * torch.eye(n, device=K.device, dtype=K.dtype)

    # Compute log-determinant
    return torch.logdet(K).item()


def _compute_coverage(
    requirement_embeddings: torch.Tensor,
    all_evidence: torch.Tensor,
    selected_evidence: torch.Tensor,
) -> tuple[float, float, float]:
    """
    Compute ideal mass, captured mass, and coverage ratio.

    :param requirement_embeddings: (R, D) tensor of requirement embeddings
    :param all_evidence: (N, D) tensor of all evidence embeddings
    :param selected_evidence: (S, D) tensor of selected evidence embeddings (may be empty)
    :return: (ideal_mass, captured_mass, coverage_ratio)
    """
    # Ideal: max similarity to all evidence for each requirement
    sim_all = _compute_cosine_similarity(requirement_embeddings, all_evidence)
    ideal_per_req, _ = sim_all.max(dim=1)
    ideal_mass = ideal_per_req.sum().item()

    # Captured: max similarity to selected evidence for each requirement
    if selected_evidence.shape[0] == 0:
        captured_mass = 0.0
    else:
        sim_selected = _compute_cosine_similarity(requirement_embeddings, selected_evidence)
        captured_per_req, _ = sim_selected.max(dim=1)
        captured_mass = captured_per_req.sum().item()

    coverage = captured_mass / ideal_mass if ideal_mass > 0 else 0.0
    return ideal_mass, captured_mass, coverage


class HierarchicalCoverageEvaluator:
    """
    Evaluates how well selected evidence satisfies paper requirements
    compared to the theoretical best possible retrieval.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.counter = AggregateMetricsCounter()

    def evaluate(
        self,
        paper_data: Dict,
        selected_indices: List[int],
        all_evidence: torch.Tensor,
        selected_phi: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Evaluate coverage of selected evidence against paper requirements.

        :param paper_data: Nested dictionary from .pt file with 'general_requirements'
                          and 'specific_requirements' keys.
        :param selected_indices: Indices of chunks chosen by the solver.
        :param all_evidence: Tensor (N, D) of ALL chunks in the paper.
        :param selected_phi: Tensor (S,) of phi/importance values for selected items.
        :return: Dictionary of evaluation metrics.
        """
        device = all_evidence.device

        # Extract selected evidence embeddings
        selected_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=device)
        selected_evidence = all_evidence[selected_indices_tensor]

        # A. General Requirements Evaluation
        gr_embeddings = paper_data["general_requirements"]["embeddings"].to(device)
        _, _, gr_coverage = _compute_coverage(gr_embeddings, all_evidence, selected_evidence)

        # B. Decision Node Evaluation - report each node's coverage
        metrics: Dict[str, float] = {"global_gr_coverage": gr_coverage}
        decision_coverages: List[float] = []

        for decision_node, sr_section in paper_data["specific_requirements"].items():
            sr_embeddings = sr_section["embeddings"].to(device)
            _, _, node_coverage = _compute_coverage(sr_embeddings, all_evidence, selected_evidence)
            decision_coverages.append(node_coverage)
            metrics[f"decision_{decision_node}_coverage"] = node_coverage

        avg_decision_coverage = sum(decision_coverages) / len(decision_coverages) if decision_coverages else 0.0
        metrics["avg_decision_coverage"] = avg_decision_coverage

        # C. Selection Quality Metrics
        # 1. Set Redundancy (lower is better)
        metrics["redundancy_score"] = _compute_set_redundancy(selected_evidence)

        # 2. Information Density (higher is better)
        if selected_phi is not None:
            metrics["avg_density_phi"] = _compute_information_density(selected_phi)
        else:
            metrics["avg_density_phi"] = 0.0

        # 3. Semantic Volume / Diversity (higher is better)
        metrics["semantic_volume"] = _compute_semantic_volume(selected_evidence)

        # Update aggregate counter
        self.counter.update(metrics)

        return metrics
