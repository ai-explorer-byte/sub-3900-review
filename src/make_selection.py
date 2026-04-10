from typing import Dict, List, Tuple

from src.data_loader import EvidenceUnit
from src.make_optimizer import make_optimizer


def make_selection(
    units: List[EvidenceUnit],
    config: Dict,
    objective: str,
    solver: str,
) -> Tuple[List[EvidenceUnit], List[int], int, float]:
    """
    Main retrieval loop: select evidence units under budget constraint.

    Args:
        units: List of EvidenceUnit objects with importance and cost populated
        config: Configuration dictionary
        objective: One of "mmr", "graphcut", "fl", "dpp"
        solver: One of "greedy", "cost_norm", "dpp"

    Returns:
        Tuple of:
            - List of selected EvidenceUnit objects
            - List of selected indices
            - Total cost used
            - Total importance captured
    """
    # Handle DPP as both objective and solver
    if objective == "dpp":
        solver = "dpp"

    objective_fn, solver_instance = make_optimizer(config, objective, solver)

    selected_indices, total_cost = solver_instance.select(units, objective_fn)

    selected_units = [units[i] for i in selected_indices]
    total_importance = sum(u.importance for u in selected_units)

    return selected_units, selected_indices, total_cost, total_importance
