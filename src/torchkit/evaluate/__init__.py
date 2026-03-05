from torchkit.evaluate._evaluator import CompositeEvaluator, Evaluator

from torchkit.evaluate.classification_evaluator import ClassificationEvaluator
from torchkit.evaluate.regression_evaluator import RegressionEvaluator
from torchkit.evaluate.segmentation_evaluator import SegmentationEvaluator, Segmentation3DEvaluator

from torchkit.evaluate.roc_evaluator import ROCBinaryEvaluator

from torchkit.evaluate.calibration_evaluator import CalibrationEvaluator
from torchkit.evaluate.dca_evaluator import DCAEvaluator

__all__ = [
    Evaluator,
    CompositeEvaluator,
    ClassificationEvaluator,
    RegressionEvaluator,
    SegmentationEvaluator,
    Segmentation3DEvaluator,
    CalibrationEvaluator,
    DCAEvaluator,
    ROCBinaryEvaluator
]