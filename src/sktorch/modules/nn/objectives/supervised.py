# some supervised objectives

"""Implemented Supervised Objectives:
 + CrossEntropyLoss
 + MSELoss
 + More to come...
"""

from sktorch.modules.nn.objectives._base import LossOut, SupervisedObjective
import torch.nn.functional as F

class CrossEntropyLoss(SupervisedObjective):
    """
    Classification Loss.\\
    Computes Cross-Entropy Loss between `out['clf/logits']` and `out['clf/targets']`
    """
    def __init__(self, name: str = "cross_entropy_loss", required: bool = True):
        super().__init__(
            name=name,
            required=required,
            required_pred_keys=('clf/logits',),
            required_target_keys=('clf/targets',),
        )
    
    def loss(self, predictions, targets):
        """Computes Cross-Entropy Loss between `out['clf/logits']` and `out['clf/targets']`
        """
        
        logits = predictions['clf/logits']
        labels = targets['clf/targets']
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return LossOut(loss=loss, details={})
    

class MSELoss(SupervisedObjective):
    """
    Regression Loss.\\
    Computes MSELoss between `out['reg/pred']` and `out['reg/target']`
    """
    def __init__(self, name: str = "mse_loss", required: bool = True):
        super().__init__(
            name=name,
            required=required,
            required_pred_keys=('reg/pred',),
            required_target_keys=('reg/target',),
        )
    
    def loss(self, predictions, targets):
        """Computes MSE between `out['reg/pred']` and `out['reg/target']`
        """
        
        preds = predictions['reg/pred']
        tgts = targets['reg/target']
        loss = F.mse_loss(preds, tgts, reduction='mean')
        return LossOut(loss=loss, details={})