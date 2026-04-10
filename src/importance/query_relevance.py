import logging
from typing import Dict, List
import torch
import torch.nn.functional as F

from src.data_loader import EvidenceUnit

logger = logging.getLogger(__name__)


class QueryRelevanceScorer:
    def __init__(self, config: Dict):
        self.config = config

    def score(
        self, units: List[EvidenceUnit], query_embedding: torch.Tensor
    ) -> List[EvidenceUnit]:
        embeddings = torch.stack([u.embedding for u in units])

        # Check for NaN values and filter out affected units
        nan_mask = torch.isnan(embeddings).any(dim=1)
        if nan_mask.any():
            nan_indices = torch.where(nan_mask)[0].tolist()
            for idx in nan_indices:
                unit = units[idx]
                logger.warning(f"NaN embedding detected for unit index {idx}: {getattr(unit, 'id', 'unknown')}")
            logger.warning(f"Removing {len(nan_indices)} units with NaN embeddings from {len(units)} total")

            # Filter out units with NaN embeddings
            valid_mask = ~nan_mask
            units = [u for i, u in enumerate(units) if valid_mask[i]]
            embeddings = embeddings[valid_mask]

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        query_norm = F.normalize(query_embedding.unsqueeze(0), p=2, dim=1)

        # Cosine similarity: (n,)
        similarities = (embeddings_norm @ query_norm.T).squeeze(1)

        for i, unit in enumerate(units):
            unit.importance = similarities[i].item()

        return units
