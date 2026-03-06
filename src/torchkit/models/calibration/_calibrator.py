from __future__ import annotations

from torch import nn, Tensor
import torch

class Calibrator(nn.Module):
    # subclass of nn.Module does not necessarily mean neural network.
    # it gives standard interface: forward, state_dict, to(device) etc.
    # during implementation, if e.g., temperature scaling, have a single parameter and handle its learning during fit.

    # contracts for calibration: takes the TaskHead output ModelOut["logits"] and batch["y/target/targets/label/labels"]
    # TODO: verify the contracts and define the Calibrator interface. Implement `forward_impl` and leave `forward` for interface validation.
    pass