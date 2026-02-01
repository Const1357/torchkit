from abc import abstractmethod

from torch import Tensor
from numpy import ndarray
import numpy as np



class Evaluator:
    
    def __init__(self):
        pass

    def evaluate(self, predictions: Tensor | ndarray, targets: Tensor | ndarray) -> dict:
        """Evaluate the predictions against the targets.

        Args:
            predictions: The predicted values.
            targets: The ground truth values.

        Returns:
            Dictionary of evaluation metrics.
        """

        # ensure both inputs are of the same shape
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} and targets shape {targets.shape} must be the same.")


        if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
            return self._evaluate_numpy(predictions, targets)
        elif isinstance(predictions, Tensor) and isinstance(targets, Tensor):
            return self._evaluate_torch(predictions, targets)
        else:
            raise TypeError("Predictions and targets must be either both numpy arrays or both PyTorch tensors.")

    @abstractmethod
    def _evaluate_numpy(self, predictions: ndarray, targets: ndarray) -> dict:
        """Evaluate using numpy arrays.

        Args:
            predictions: The predicted values as numpy arrays.
            targets: The ground truth values as numpy arrays.

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    @abstractmethod
    def _evaluate_torch(self, predictions: Tensor, targets: Tensor) -> dict:
        """Evaluate using PyTorch tensors.

        Args:
            predictions: The predicted values as PyTorch tensors.
            targets: The ground truth values as PyTorch tensors.

        Returns:
            Dictionary of evaluation metrics.
        """
        if (predictions.device != targets.device):
            raise ValueError(f"Predictions ({predictions.device}) and targets ({targets.device}) must be on the same device.")
        pass

class RegressionEvaluator(Evaluator):

    def __init__(self):
        super().__init__()

    def evaluate(self, predictions, targets):
        """Evaluate the predictions against the targets.

        Args:
            predictions: The predicted values.
            targets: The ground truth values.

        Returns:
            #TODO: Update this docstring
        """
        return super().evaluate(predictions, targets)
    
    def _evaluate_numpy(self, predictions: Tensor | ndarray, targets: Tensor | ndarray) -> dict:
        # Implementation of regression evaluation metrics
        # R^2, MAE, RMSE, etc.
        pass

class ClassificationEvaluator(Evaluator):
    
    def _evaluate_numpy(self, predictions: Tensor | ndarray, targets: Tensor | ndarray) -> dict:
        # Implementation of classification evaluation metrics
        # Accuracy, Precision, Recall, F1-score, AUC, etc.
        pass

class SegmentationEvaluator(Evaluator):
    
    def _evaluate_numpy(self, predictions: Tensor | ndarray, targets: Tensor | ndarray) -> dict:
        # Implementation of segmentation evaluation metrics
        # Dice coefficient, Jaccard index, Hausdorff distance, etc.
        pass