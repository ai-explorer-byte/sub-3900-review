import math
import random
from typing import Dict, Optional
import torch


class PenaltyCostProfiler:
    """
    Penalty Model: Cost = Tokens * (1 + gamma * (1 - phi))
    Model: "davanstrien/ModernBERT-based-Reasoning-Required"
    """

    def __init__(self, config: Dict, device: str = "cpu"):
        self.gamma = config["gamma"]
        self.device = device
        self.model_name = "davanstrien/ModernBERT-based-Reasoning-Required"
        self._tokenizer = None
        self._model = None
        self._cache: Dict[str, float] = {}

    def _ensure_model_loaded(self):
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = (
                AutoModelForSequenceClassification.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )

    def _get_phi(self, text: Optional[str]) -> float:
        """
        Calculates Semantic Density (phi) from logits.
        1.0 = High Density, 0.0 = Fluff.
        """
        # Mock for .pt data (No text available)
        if text is None:
            return random.uniform(0.0, 1.0)

        if text in self._cache:
            return self._cache[text]

        self._ensure_model_loaded()
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze()

        # LOGITS TO PHI NORMALIZATION
        if logits.numel() == 1:
            # Regression: Sigmoid
            val = float(logits.item())
            phi = 1.0 / (1.0 + math.exp(-val))
        else:
            # Classification: Normalized Expected Value
            probs = torch.softmax(logits.float(), dim=0)
            C = int(probs.numel())
            # weighted sum of class indices (0..C-1)
            expected = float(
                (probs * torch.arange(C, device=probs.device)).sum().item()
            )
            # Normalize to [0, 1]
            phi = expected / max(1.0, float(C - 1))

        phi = max(0.0, min(1.0, phi))
        self._cache[text] = phi
        return phi

    def calculate_cost(self, tokens: int, text: Optional[str] = None) -> int:
        phi = self._get_phi(text)

        # PENALTY FORMULA
        # If phi=1.0 (dense), multiplier=1.0
        # If phi=0.0 (fluff), multiplier=1.0 + gamma
        multiplier = 1.0 + (self.gamma * (1.0 - phi))

        return int(round(tokens * multiplier))
