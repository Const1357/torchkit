import torch
from sktorch.modules.nn.models._base._estimator import SKTorchEstimatorBase
from sktorch.modules.nn.objectives.composite import Objective
from typing import Any, Dict, Optional

from sklearn.base import clone

import optuna
        
