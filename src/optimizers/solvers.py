from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Set, Tuple
import torch
import torch.nn.functional as F

from src.data_loader import EvidenceUnit
from .objectives import _get_similarity_matrix


class BaseSolver(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.budget = config["budget_B"]

    @abstractmethod
    def select(
        self,
        units: List[EvidenceUnit],
        gain_fn: Callable[[int, Set[int], List[EvidenceUnit], torch.Tensor, Dict], float],
    ) -> Tuple[List[int], int]:
        pass


class GreedySolver(BaseSolver):
    """
    Standard greedy: Select argmax gain(u) subject to budget constraint.
    """

    def select(
        self,
        units: List[EvidenceUnit],
        gain_fn: Callable[[int, Set[int], List[EvidenceUnit], torch.Tensor, Dict], float],
    ) -> Tuple[List[int], int]:
        sim_matrix = _get_similarity_matrix(units)
        selected: List[int] = []
        selected_set: Set[int] = set()
        total_cost = 0

        candidates = set(range(len(units)))

        while candidates:
            best_idx = -1
            best_gain = float("-inf")

            for idx in candidates:
                if total_cost + units[idx].cost > self.budget:
                    continue
                gain = gain_fn(idx, selected_set, units, sim_matrix, self.config)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            total_cost += units[best_idx].cost
            candidates.remove(best_idx)

        return selected, total_cost


class CostNormalizedSolver(BaseSolver):
    """
    Cost-normalized greedy: Select argmax gain(u) / cost(u).
    """

    def select(
        self,
        units: List[EvidenceUnit],
        gain_fn: Callable[[int, Set[int], List[EvidenceUnit], torch.Tensor, Dict], float],
    ) -> Tuple[List[int], int]:
        sim_matrix = _get_similarity_matrix(units)
        selected: List[int] = []
        selected_set: Set[int] = set()
        total_cost = 0

        candidates = set(range(len(units)))

        while candidates:
            best_idx = -1
            best_ratio = float("-inf")

            for idx in candidates:
                cost = units[idx].cost
                if total_cost + cost > self.budget:
                    continue
                if cost <= 0:
                    continue
                gain = gain_fn(idx, selected_set, units, sim_matrix, self.config)
                ratio = gain / cost
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_idx = idx

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            total_cost += units[best_idx].cost
            candidates.remove(best_idx)

        return selected, total_cost


class TopkSolver(BaseSolver):
    """
    Simple top-k selection: Sort units by importance and select until budget is exhausted.
    No gain function or diversity consideration - just picks highest importance units.
    """

    def select(
        self,
        units: List[EvidenceUnit],
        gain_fn: Callable[[int, Set[int], List[EvidenceUnit], torch.Tensor, Dict], float],
    ) -> Tuple[List[int], int]:
        # Sort indices by importance (descending)
        sorted_indices = sorted(range(len(units)), key=lambda i: units[i].importance, reverse=True)

        selected: List[int] = []
        total_cost = 0

        for idx in sorted_indices:
            if total_cost + units[idx].cost > self.budget:
                continue
            selected.append(idx)
            total_cost += units[idx].cost

        return selected, total_cost


class DPPSolver(BaseSolver):
    """
    DPP-based selection using L-ensemble kernel.
    L = diag(q) @ S @ diag(q) where q is quality (importance) and S is similarity.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.quality_weight = config.get("dpp_quality_weight", 1.0)

    def _build_L_matrix(self, units: List[EvidenceUnit]) -> torch.Tensor:
        embeddings = torch.stack([u.embedding for u in units])
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        S = embeddings_norm @ embeddings_norm.T

        # Quality vector from importance scores
        q = torch.tensor([u.importance for u in units])
        q = q.clamp(min=1e-6)  # Ensure positive
        q = q ** self.quality_weight

        # L = diag(q) @ S @ diag(q)
        L = torch.diag(q) @ S @ torch.diag(q)
        return L

    def select(
        self,
        units: List[EvidenceUnit],
        gain_fn: Callable[[int, Set[int], List[EvidenceUnit], torch.Tensor, Dict], float],
    ) -> Tuple[List[int], int]:
        from .objectives import dpp_log_det_gain

        L_matrix = self._build_L_matrix(units)
        selected: List[int] = []
        selected_set: Set[int] = set()
        total_cost = 0

        candidates = set(range(len(units)))

        while candidates:
            best_idx = -1
            best_gain = float("-inf")

            for idx in candidates:
                if total_cost + units[idx].cost > self.budget:
                    continue
                gain = dpp_log_det_gain(idx, selected_set, units, L_matrix, self.config)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx == -1:
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            total_cost += units[best_idx].cost
            candidates.remove(best_idx)

        return selected, total_cost
