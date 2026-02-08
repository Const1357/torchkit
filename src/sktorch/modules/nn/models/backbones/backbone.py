from dataclasses import dataclass, field
from typing import Dict, Any
from torch import Tensor


@dataclass(frozen=True)
class BackboneOut:
    """
    Output of a backbone.

    Attributes
    ----------
    features : Tensor | Dict[str, Tensor]
        Extracted features.
        - Tensor: single feature representation (single-task backbones).
        - Dict[str, Tensor]: multiple named feature maps (multitask or multi-endpoint backbones).
          Must only be used when the consuming estimator explicitly supports feature routing.
    details : Dict[str, Any]
        Optional auxiliary information (metadata, diagnostics).
    """
    features: Tensor | Dict[str, Tensor]
    details: Dict[str, Any] = field(default_factory=dict)