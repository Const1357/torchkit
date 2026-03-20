from __future__ import annotations

from torch import nn

from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelFactory, TorchkitModelSpec
from torchkit.models.adapters._feature_adapter import GAPAdapter
from torchkit.models.adapters.factory import FeatureAdapterFactory, FeatureAdapterSpec
from torchkit.models.backbone.MLP_backbone import MLPBackbone
from torchkit.models.backbone.factory import BackboneFactory, BackboneSpec
from torchkit.models.calibration.factory import CalibratorFactory, CalibratorSpec
from torchkit.models.calibration.temperature import TemperatureScalingCalibrator
from torchkit.models.decision.classification import BinaryClassificationThreshold
from torchkit.models.decision.factory import DecisionModuleFactory, DecisionModuleSpec
from torchkit.models.fuse._fuse_module import ConcatFuseModule
from torchkit.models.fuse.factory import FuseModuleFactory, FuseModuleSpec
from torchkit.models.head._task_head import TaskHead
from torchkit.models.head.factory import TaskHeadSpec
from torchkit.models.head_module.classification import ClassifierHeadMLP
from torchkit.models.head_module.factory import HeadModuleFactory, HeadModuleSpec
from torchkit.models.prediction._prediction_head import PredictionHead
from torchkit.models.prediction.factory import PredictionHeadSpec
from torchkit.models.probability_mapping.classification import ClassificationProbabilityMapper
from torchkit.models.probability_mapping.factory import (
    ProbabilityMapperFactory,
    ProbabilityMapperSpec,
)


def test_leaf_modules_to_spec_round_trip():
    backbone = MLPBackbone(input_dim=3, hidden_dims=[5], output_dim=2, dropout=0.1)
    backbone_spec = backbone.to_spec()
    assert isinstance(backbone_spec, BackboneSpec)
    assert backbone_spec.cls is MLPBackbone
    assert backbone_spec.kwargs["input_dim"] == 3
    assert backbone_spec.kwargs["hidden_dims"] == [5]
    assert backbone_spec.kwargs["output_dim"] == 2
    assert backbone_spec.kwargs["dropout"] == 0.1
    assert isinstance(BackboneFactory.build(backbone_spec), MLPBackbone)

    adapter = GAPAdapter(keepdim=True)
    adapter_spec = adapter.to_spec()
    assert isinstance(adapter_spec, FeatureAdapterSpec)
    assert adapter_spec.cls is GAPAdapter
    assert adapter_spec.kwargs == {"keepdim": True}
    assert isinstance(FeatureAdapterFactory.build(adapter_spec), GAPAdapter)

    fuse_module = ConcatFuseModule(dim=2)
    fuse_spec = fuse_module.to_spec()
    assert isinstance(fuse_spec, FuseModuleSpec)
    assert fuse_spec.cls is ConcatFuseModule
    assert fuse_spec.kwargs == {"dim": 2}
    assert isinstance(FuseModuleFactory.build(fuse_spec), ConcatFuseModule)

    head_module = ClassifierHeadMLP(
        hidden_dims=[4],
        num_classes=3,
        input_dim=6,
        activation=nn.LeakyReLU,
        dropout=0.2,
    )
    head_module_spec = head_module.to_spec()
    assert isinstance(head_module_spec, HeadModuleSpec)
    assert head_module_spec.cls is ClassifierHeadMLP
    assert head_module_spec.kwargs["hidden_dims"] == [4]
    assert head_module_spec.kwargs["num_classes"] == 3
    assert head_module_spec.kwargs["input_dim"] == 6
    assert head_module_spec.kwargs["activation"] is nn.LeakyReLU
    assert head_module_spec.kwargs["dropout"] == 0.2
    assert isinstance(HeadModuleFactory.build(head_module_spec), ClassifierHeadMLP)

    calibrator = TemperatureScalingCalibrator(init_temp=2.5, max_iter=7, lr=0.02, active=True)
    calibrator_spec = calibrator.to_spec()
    assert isinstance(calibrator_spec, CalibratorSpec)
    assert calibrator_spec.cls is TemperatureScalingCalibrator
    assert calibrator_spec.kwargs == {"init_temp": 2.5, "max_iter": 7, "lr": 0.02}
    assert calibrator_spec.active is True
    assert isinstance(CalibratorFactory.build(calibrator_spec), TemperatureScalingCalibrator)

    decision_module = BinaryClassificationThreshold(threshold=0.7)
    decision_spec = decision_module.to_spec()
    assert isinstance(decision_spec, DecisionModuleSpec)
    assert decision_spec.cls is BinaryClassificationThreshold
    assert decision_spec.kwargs == {"threshold": 0.7}
    assert isinstance(DecisionModuleFactory.build(decision_spec), BinaryClassificationThreshold)

    probability_mapper = ClassificationProbabilityMapper()
    probability_mapper_spec = probability_mapper.to_spec()
    assert isinstance(probability_mapper_spec, ProbabilityMapperSpec)
    assert probability_mapper_spec.cls is ClassificationProbabilityMapper
    assert probability_mapper_spec.kwargs == {}
    assert isinstance(
        ProbabilityMapperFactory.build(probability_mapper_spec),
        ClassificationProbabilityMapper,
    )


def test_nested_model_to_spec_round_trip():
    model = TorchkitModel(
        backbone=MLPBackbone(input_dim=3, hidden_dims=[5], output_dim=4),
        heads={
            "clf": TaskHead(
                required_features="features",
                feature_adapter=GAPAdapter(keepdim=False),
                head_module=ClassifierHeadMLP(
                    hidden_dims=[6],
                    num_classes=2,
                    input_dim=4,
                ),
                active=False,
            )
        },
        prediction_heads={
            "clf": PredictionHead(
                calibrator=TemperatureScalingCalibrator(init_temp=1.5, active=True),
                probability_mapper=ClassificationProbabilityMapper(),
                decision_module=BinaryClassificationThreshold(threshold=0.6),
                active=False,
            )
        },
    )

    spec = model.to_spec()

    assert isinstance(spec, TorchkitModelSpec)
    assert isinstance(spec.backbone, BackboneSpec)
    assert spec.backbone.cls is MLPBackbone

    assert set(spec.heads.keys()) == {"clf"}
    head_spec = spec.heads["clf"]
    assert isinstance(head_spec, TaskHeadSpec)
    assert head_spec.required_features == "features"
    assert head_spec.active is False
    assert isinstance(head_spec.feature_adapter, FeatureAdapterSpec)
    assert head_spec.feature_adapter.cls is GAPAdapter
    assert head_spec.feature_adapter.kwargs == {"keepdim": False}
    assert isinstance(head_spec.head_module, HeadModuleSpec)
    assert head_spec.head_module.cls is ClassifierHeadMLP
    assert head_spec.head_module.kwargs["hidden_dims"] == [6]
    assert head_spec.head_module.kwargs["num_classes"] == 2
    assert head_spec.head_module.kwargs["input_dim"] == 4

    assert spec.prediction_heads is not None
    prediction_head_spec = spec.prediction_heads["clf"]
    assert isinstance(prediction_head_spec, PredictionHeadSpec)
    assert prediction_head_spec.active is False
    assert isinstance(prediction_head_spec.calibrator, CalibratorSpec)
    assert prediction_head_spec.calibrator.cls is TemperatureScalingCalibrator
    assert prediction_head_spec.calibrator.kwargs["init_temp"] == 1.5
    assert prediction_head_spec.calibrator.active is True
    assert isinstance(prediction_head_spec.probability_mapper, ProbabilityMapperSpec)
    assert prediction_head_spec.probability_mapper.cls is ClassificationProbabilityMapper
    assert isinstance(prediction_head_spec.decision_module, DecisionModuleSpec)
    assert prediction_head_spec.decision_module.cls is BinaryClassificationThreshold
    assert prediction_head_spec.decision_module.kwargs["threshold"] == 0.6

    rebuilt = TorchkitModelFactory.build(spec)
    assert isinstance(rebuilt, TorchkitModel)
    assert isinstance(rebuilt.backbone, MLPBackbone)
    assert isinstance(rebuilt.heads["clf"].feature_adapter, GAPAdapter)
    assert isinstance(rebuilt.heads["clf"].head_module, ClassifierHeadMLP)
    assert rebuilt.heads["clf"].is_active is False
    assert isinstance(rebuilt.prediction_heads["clf"].calibrator, TemperatureScalingCalibrator)
    assert rebuilt.prediction_heads["clf"].calibrator.is_active is True
    assert isinstance(
        rebuilt.prediction_heads["clf"].probability_mapper,
        ClassificationProbabilityMapper,
    )
    assert isinstance(
        rebuilt.prediction_heads["clf"].decision_module,
        BinaryClassificationThreshold,
    )
    assert rebuilt.prediction_heads["clf"].is_active is False
