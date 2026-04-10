from typing import Callable, Dict, List, Set, Tuple
import torch

from src.data_loader import EvidenceUnit
from src.optimizers import (
    mmr_gain,
    graphcut_gain,
    facility_location_gain,
    GreedySolver,
    CostNormalizedSolver,
    DPPSolver,
    TopkSolver,
)
from src.optimizers.solvers import BaseSolver


def get_objective_fn(
    objective: str,
) -> Callable[[int, Set[int], List[EvidenceUnit], torch.Tensor, Dict], float]:
    """
    Returns the marginal gain function for the specified objective.
    """
    objectives = {
        "mmr": mmr_gain,
        "graphcut": graphcut_gain,
        "fl": facility_location_gain,
    }
    assert objective in objectives, f"Unknown objective: {objective}. Must be one of: {list(objectives.keys())}"
    return objectives[objective]


def get_solver(solver: str, config: Dict) -> BaseSolver:
    """
    Returns the solver instance for the specified solver type.
    """
    solvers = {
        "greedy": GreedySolver,
        "cost_norm": CostNormalizedSolver,
        "dpp": DPPSolver,
        "topk": TopkSolver,
    }
    assert solver in solvers, f"Unknown solver: {solver}. Must be one of: {list(solvers.keys())}"
    return solvers[solver](config)


def make_optimizer(
    config: Dict, objective: str, solver: str
) -> Tuple[Callable, BaseSolver]:
    """
    Creates and returns the objective function and solver.

    Args:
        config: Configuration dictionary
        objective: One of "mmr", "graphcut", "fl"
        solver: One of "greedy", "cost_norm", "dpp", "topk"

    Returns:
        Tuple of (objective_fn, solver_instance)
    """
    # DPP uses its own internal objective, topk ignores objective entirely
    if solver in ("dpp", "topk"):
        return None, get_solver(solver, config)

    objective_fn = get_objective_fn(objective)
    solver_instance = get_solver(solver, config)
    return objective_fn, solver_instance
