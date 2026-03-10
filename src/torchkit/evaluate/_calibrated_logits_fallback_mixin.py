from __future__ import annotations

from typing import Any

import torch


class CalibratedLogitsFallbackMixin:
    """
    Mixin for evaluators whose `score_key` may point to `.../calibrated_logits`,
    with fallback to `.../logits` when calibrated logits are unavailable.

    Intended behavior:
    - Validation pass accepts raw logits when `score_key` is calibrated logits and it's not available.
    - Metric computation tries calibrated logits first, then falls back to raw logits.
    """

    score_key: str

    def _fallback_score_key(self) -> str | None:
        if self.score_key.endswith("calibrated_logits"):
            return self.score_key[:-len("calibrated_logits")] + "logits"
        return None

    def _resolve_score_tensor(self, inputs: dict[str, Any]) -> torch.Tensor:
        try:
            return self.resolve(inputs, self.score_key).detach()
        except KeyError:
            fallback_key = self._fallback_score_key()
            if fallback_key is None:
                raise
            return self.resolve(inputs, fallback_key).detach()

    def _validation_required_keys(self) -> tuple[str, ...]:
        # overrides Evaluator._validation_required_keys to allow fallback of calibrated_logits to logits
        fallback_key = self._fallback_score_key()
        if fallback_key is None:
            return self.required_keys

        remapped_keys: list[str] = []
        for key in self.required_keys:
            if key == self.score_key:
                remapped_keys.append(fallback_key)
            else:
                remapped_keys.append(key)

        return tuple(remapped_keys)