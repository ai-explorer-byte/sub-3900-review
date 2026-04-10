import logging
from typing import Dict, List
import torch
import torch.nn.functional as F

from src.data_loader import EvidenceUnit
from .importance_prior import ImportancePriorScorer

logger = logging.getLogger(__name__)


class QueryAugmentedScorer:
    def __init__(self, config: Dict):
        self.config = config
        self.prior_scorer = ImportancePriorScorer(config)

    def score(
        self, units: List[EvidenceUnit], query_embedding: torch.Tensor
    ) -> List[EvidenceUnit]:
        # Get prior importance scores
        units = self.prior_scorer.score(units)
        prior_scores = torch.tensor([u.importance for u in units])

        # Compute query relevance (NaN units already filtered by prior_scorer)
        embeddings = torch.stack([u.embedding for u in units])
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)
        query_sims = (embeddings_norm @ query_norm.T).squeeze(1)

        # Hybrid: importance weighted by query relevance
        # Normalize both to [0,1] range for combination
        prior_min, prior_max = prior_scores.min(), prior_scores.max()
        if prior_max > prior_min:
            prior_norm = (prior_scores - prior_min) / (prior_max - prior_min)
        else:
            prior_norm = torch.ones_like(prior_scores)

        query_min, query_max = query_sims.min(), query_sims.max()
        if query_max > query_min:
            query_norm_scores = (query_sims - query_min) / (query_max - query_min)
        else:
            query_norm_scores = torch.ones_like(query_sims)

        # Combine: geometric mean
        combined = torch.sqrt(prior_norm * query_norm_scores)

        for i, unit in enumerate(units):
            unit.importance = combined[i].item()

        return units
