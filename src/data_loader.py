from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
import torch


@dataclass
class EvidenceUnit:
    unit_id: str
    source_type: str  # 'paper', 'general_requirements', 'specific_requirements'
    embedding: torch.Tensor
    tokens: int = 0
    cost: int = 0
    importance: float = 0.0
    text: Optional[str] = None
    decision_node: Optional[str] = None  # For specific_requirements
    metadata: Dict = field(default_factory=dict)


def load_embeddings(config: Dict) -> Dict:
    data_path = config["data_path"]
    return torch.load(data_path, map_location="cpu", weights_only=False)


def flatten_paper_data(
    paper_data: Dict, config: Dict, paper_id: str
) -> List[EvidenceUnit]:
    units: List[EvidenceUnit] = []
    mock_min = config["mock_token_min"]
    mock_max = config["mock_token_max"]

    # Source A: Main text (paper)
    if "paper" in paper_data:
        paper_section = paper_data["paper"]
        ids = paper_section["ids"]
        embeddings = paper_section["embeddings"]
        assert len(ids) == embeddings.shape[0], f"Mismatch in paper ids vs embeddings for {paper_id}"
        for i, chunk_id in enumerate(ids):
            units.append(EvidenceUnit(
                unit_id=chunk_id,
                source_type="paper",
                embedding=embeddings[i],
                tokens=random.randint(mock_min, mock_max),
                text=None,
            ))

    # Source B: General requirements
    if "general_requirements" in paper_data:
        gr_section = paper_data["general_requirements"]
        keys = gr_section["keys"]
        embeddings = gr_section["embeddings"]
        assert len(keys) == embeddings.shape[0], f"Mismatch in general_requirements keys vs embeddings for {paper_id}"
        for i, key in enumerate(keys):
            units.append(EvidenceUnit(
                unit_id=key,
                source_type="general_requirements",
                embedding=embeddings[i],
                tokens=random.randint(mock_min, mock_max),
                text=None,
            ))

    # Source C: Specific requirements (nested)
    if "specific_requirements" in paper_data:
        for decision_node, sr_section in paper_data["specific_requirements"].items():
            keys = sr_section["keys"]
            embeddings = sr_section["embeddings"]
            assert len(keys) == embeddings.shape[0], f"Mismatch in specific_requirements/{decision_node} for {paper_id}"
            for i, key in enumerate(keys):
                units.append(EvidenceUnit(
                    unit_id=key,
                    source_type="specific_requirements",
                    embedding=embeddings[i],
                    tokens=random.randint(mock_min, mock_max),
                    text=None,
                    decision_node=decision_node,
                ))

    return units


def get_paper_units(
    data: Dict, model_name: str, paper_id: str, config: Dict
) -> List[EvidenceUnit]:
    model_data = data[model_name]
    paper_data = model_data[paper_id]
    return flatten_paper_data(paper_data, config, paper_id)


def list_papers(data: Dict, model_name: str) -> List[str]:
    return list(data[model_name].keys())


def list_models(data: Dict) -> List[str]:
    return list(data.keys())
