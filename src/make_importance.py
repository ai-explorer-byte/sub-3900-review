from typing import Dict, List, Optional
import torch

from src.data_loader import EvidenceUnit
from src.importance import ImportancePriorScorer, CoresetImportanceScorer, QueryRelevanceScorer, QueryAugmentedScorer


def make_importance(
    units: List[EvidenceUnit],
    config: Dict,
    strategy: str = "prior",
    query_embedding: Optional[torch.Tensor] = None,
) -> List[EvidenceUnit]:
    """
    Score units using the specified strategy.

    Args:
        units: List of EvidenceUnit objects
        config: Configuration dictionary
        strategy: One of "prior", "relevance", "augmented"
        query_embedding: Required for "relevance" and "augmented" strategies

    Returns:
        List of EvidenceUnit objects with importance scores populated
    """
    if strategy == "prior":
        prior_method = config.get("prior_method", "kmeans")
        if prior_method == "kmeans":
            scorer = ImportancePriorScorer(config)
        elif prior_method == "coreset":
            scorer = CoresetImportanceScorer(config)
        else:
            raise ValueError(f"Unknown prior_method: {prior_method}. Must be one of: kmeans, coreset")
        return scorer.score(units)

    elif strategy == "relevance":
        assert query_embedding is not None, "query_embedding required for relevance strategy"
        scorer = QueryRelevanceScorer(config)
        return scorer.score(units, query_embedding)

    elif strategy == "augmented":
        assert query_embedding is not None, "query_embedding required for augmented strategy"
        scorer = QueryAugmentedScorer(config)
        return scorer.score(units, query_embedding)

    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be one of: prior, relevance, augmented")
