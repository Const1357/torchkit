from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

class Evaluator(ABC):
    """
    Shared abstract base for all evaluators.

    Common responsibilities:
    - name handling
    - nested key resolution via "/" paths
    - required / optional key validation
    - validation override hook for fallback-aware evaluators
      (e.g. calibrated logits fallback to logits)

    Concrete layers:
    - ReportEvaluator: returns dict[str, Any]
    - SelectorEvaluator: returns one scalar Tensor
    """

    def __init__(self, *, name: str) -> None:
        super().__init__()

        if not isinstance(name, str) or not name:
            raise ValueError("Evaluator `name` must be a non-empty string.")

        self._name = name

    # properties
    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def required_keys(self) -> tuple[str, ...]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must define `required_keys`."
        )

    @property
    def optional_keys(self) -> tuple[str, ...]:
        """
        Optional keys that may be present with value None.

        Contract:
        - optional keys are still expected to exist in the payload
        - if present and None, they are accepted
        """
        return tuple()

    # helpers
    @staticmethod
    def resolve(inputs: dict[str, Any], key: str, strict: bool = True) -> Tensor | None:
        """
        Resolve a "/"-separated path from a nested dict payload.

        Parameters
        ----------
        inputs:
            Nested dict payload.
        key:
            Slash-separated key path.
        strict:
            If True, resolved value must be a Tensor.
            If False, resolved value may be None.

        Returns
        -------
        Tensor | None
        """
        current: Any = inputs
        parts = [p for p in key.split("/") if p]

        for i, part in enumerate(parts):
            path = "/".join(parts[:i]) or "<root>"

            if not isinstance(current, dict):
                raise TypeError(
                    f"Expected dict at path {path}, got {type(current).__name__}."
                )

            if part not in current:
                raise KeyError(
                    f"Key {part!r} not found at path {path}. "
                    f"Available keys: {list(current.keys())}."
                )

            current = current[part]

        if not strict and current is None:
            return None

        if not isinstance(current, Tensor):
            raise TypeError(
                f"Resolved value for key {key!r} must be Tensor, got {type(current).__name__}."
            )

        return current

    def _validation_required_keys(self) -> tuple[str, ...]:
        """
        Hook for subclasses / mixins to customize validation-time required keys.

        Example:
        - allow `pred/calibrated_logits` to validate against `pred/logits`
          when calibrated logits are absent.
        """
        return self.required_keys

    def _missing_keys(
        self,
        inputs: dict[str, Any],
        required_keys: tuple[str, ...] | None = None,
    ) -> list[str]:
        missing: list[str] = []

        required = set(required_keys) if required_keys is not None else set(self.required_keys)
        optional = set(self.optional_keys)

        # Optional keys are allowed to be None, but they should still exist.
        to_check = required | optional

        for key in to_check:
            try:
                _ = self.resolve(inputs, key, strict=(key not in optional))
            except (KeyError, TypeError):
                if key in required:
                    missing.append(key)

        return missing

    def _validate_inputs(self, *, inputs: dict[str, Any], kind: str) -> None:
        if not isinstance(inputs, dict):
            raise TypeError(
                f"{kind} inputs must be dict[str, Any], got {type(inputs).__name__}."
            )

        missing = self._missing_keys(
            inputs=inputs,
            required_keys=self._validation_required_keys(),
        )
        if missing:
            raise KeyError(
                f"{kind} '{self.name}' missing required keys: {missing}. "
                f"Top-level keys: {list(inputs.keys())}."
            )
