from __future__ import annotations

import copy
import pytest
import torch
from torch import nn, Tensor

from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelFactory, TorchkitModelSpec

from torchkit.models.backbone._backbone import Backbone
from torchkit.models.backbone.factory import BackboneSpec

from torchkit.models.head._task_head import TaskHead
from torchkit.models.head.factory import TaskHeadSpec

from torchkit.models.prediction._prediction_head import PredictionHead
from torchkit.models.prediction.factory import PredictionHeadSpec

from torchkit.models.fuse._fuse_module import FuseModule
from torchkit.models.fuse.factory import FuseModuleSpec

from torchkit.models.adapters._feature_adapter import FeatureAdapter
from torchkit.models.adapters.factory import FeatureAdapterSpec

from torchkit.models.head_module.factory import HeadModuleSpec

from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.calibration.factory import CalibratorSpec

from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper
from torchkit.models.probability_mapping.factory import ProbabilityMapperSpec

from torchkit.models.decision._decision_module import DecisionModule
from torchkit.models.decision.factory import DecisionModuleSpec


# -------------------------
# Dummy components
# -------------------------

class DummyBackbone(Backbone):
    def __init__(self):
        super().__init__(supported_features=["feat_a", "feat_b"])

    def _forward_impl(self, input: dict[str, Tensor], *, requested_features=None, **kwargs) -> dict[str, Tensor]:
        x = input["x"]
        out = {}
        if "feat_a" in requested_features:
            out["feat_a"] = x + 1.0
        if "feat_b" in requested_features:
            out["feat_b"] = x + 2.0
        return out


class DummyFuse(FuseModule):
    def __init__(self):
        super().__init__()
        self.last_payload = None

    def forward(self, features: dict[str, Tensor], *, payload=None, **kwargs) -> Tensor:
        self.last_payload = payload
        return torch.cat([features[k] for k in sorted(features.keys())], dim=1)


class StatefulFuse(FuseModule):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, features: dict[str, Tensor], **kwargs) -> Tensor:
        return list(features.values())[0] * self.scale


class IdentityAdapter(FeatureAdapter):
    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features


class AddOneAdapter(FeatureAdapter):
    def __init__(self):
        super().__init__()
        self.last_payload = None

    def forward(self, features: Tensor, *, payload=None, **kwargs) -> Tensor:
        self.last_payload = payload
        return features + 1.0


class LinearLogitsHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        return {"logits": self.linear(x)}


class LinearPredictionsHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        return {"predictions": self.linear(x)}


class HeadWithPayload(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.last_payload = None

    def forward(self, x: Tensor, *, payload=None, **kwargs) -> dict[str, Tensor]:
        self.last_payload = payload
        return {"logits": self.linear(x)}


class DummyCalibrator(Calibrator):
    def forward_impl(self, logits: Tensor) -> Tensor:
        return logits + 10.0

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        pass


class DummyProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: Tensor) -> Tensor:
        if logits.ndim == 1:
            return torch.sigmoid(logits)
        if logits.ndim == 2 and logits.shape[1] == 1:
            return torch.sigmoid(logits)
        if logits.ndim == 2:
            return torch.softmax(logits, dim=1)
        raise ValueError("Unsupported logits shape.")


class DummyDecisionModule(DecisionModule):
    def forward_impl(self, probs: Tensor) -> Tensor:
        if probs.ndim == 1:
            return (probs >= 0.5).long()
        if probs.ndim == 2 and probs.shape[1] == 1:
            return (probs[:, 0] >= 0.5).long()
        if probs.ndim == 2:
            return torch.argmax(probs, dim=1)
        raise ValueError("Unsupported probability shape.")


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def x_tensor() -> Tensor:
    return torch.randn(4, 3)


@pytest.fixture
def payload(x_tensor: Tensor) -> dict[str, Tensor]:
    return {
        "x": x_tensor.clone(),
        "tabular": torch.randn(4, 2),
        "meta": torch.randn(4, 1),
    }


@pytest.fixture
def single_task_model() -> TorchkitModel:
    backbone = DummyBackbone()
    head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )
    return TorchkitModel(backbone=backbone, heads={"clf": head})


@pytest.fixture
def multitask_model() -> TorchkitModel:
    backbone = DummyBackbone()

    clf_head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )

    reg_head = TaskHead(
        required_features={"feat_a", "feat_b"},
        fuse_module=DummyFuse(),
        feature_adapter=IdentityAdapter(),
        head_module=LinearPredictionsHead(in_features=6, out_features=1),
    )

    return TorchkitModel(
        backbone=backbone,
        heads={"clf": clf_head, "reg": reg_head},
    )


@pytest.fixture
def model_with_prediction_heads() -> TorchkitModel:
    backbone = DummyBackbone()

    clf_head = TaskHead(
        required_features="feat_a",
        feature_adapter=AddOneAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )

    reg_head = TaskHead(
        required_features={"feat_a", "feat_b"},
        fuse_module=DummyFuse(),
        feature_adapter=IdentityAdapter(),
        head_module=LinearPredictionsHead(in_features=6, out_features=1),
    )

    clf_phead = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
        active=True,
    )

    return TorchkitModel(
        backbone=backbone,
        heads={"clf": clf_head, "reg": reg_head},
        prediction_heads={"clf": clf_phead},
    )


# -------------------------
# Construction and properties
# -------------------------

def test_model_requires_backbone():
    with pytest.raises(ValueError, match="`backbone` must be provided"):
        TorchkitModel(backbone=None, heads={})  # type: ignore[arg-type]


def test_model_requires_heads(single_task_model: TorchkitModel):
    with pytest.raises(ValueError, match="`heads` must be provided"):
        TorchkitModel(backbone=single_task_model.backbone, heads=None)  # type: ignore[arg-type]


def test_model_rejects_bad_backbone_type():
    with pytest.raises(TypeError, match="`backbone` must be an instance of Backbone"):
        TorchkitModel(backbone=nn.Identity(), heads={})  # type: ignore[arg-type]


def test_model_rejects_bad_heads_type(single_task_model: TorchkitModel):
    with pytest.raises(TypeError, match="`heads` must be a dict"):
        TorchkitModel(backbone=single_task_model.backbone, heads=["bad"])  # type: ignore[arg-type]


def test_model_rejects_prediction_head_without_matching_head():
    backbone = DummyBackbone()
    head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )
    phead = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
    )

    with pytest.raises(ValueError, match="does not have a corresponding head"):
        TorchkitModel(backbone=backbone, heads={"clf": head}, prediction_heads={"other": phead})


def test_model_properties(multitask_model: TorchkitModel):
    assert multitask_model.head_names == {"clf", "reg"}
    assert multitask_model.active_head_names == {"clf", "reg"}
    assert multitask_model.prediction_head_names == set()
    assert multitask_model.active_prediction_head_names == set()
    assert multitask_model.all_required_features == {"feat_a", "feat_b"}
    assert multitask_model.active_required_features == {"feat_a", "feat_b"}


# -------------------------
# Head enable / disable
# -------------------------

def test_model_enable_disable_head(multitask_model: TorchkitModel):
    multitask_model.disable_head("clf")
    assert multitask_model.active_head_names == {"reg"}

    multitask_model.enable_head("clf")
    assert multitask_model.active_head_names == {"clf", "reg"}


def test_model_enable_disable_prediction_head_follows_head(model_with_prediction_heads: TorchkitModel):
    assert model_with_prediction_heads.active_prediction_head_names == {"clf"}

    model_with_prediction_heads.disable_head("clf")
    assert model_with_prediction_heads.active_head_names == {"reg"}
    assert model_with_prediction_heads.active_prediction_head_names == set()

    model_with_prediction_heads.enable_head("clf")
    assert model_with_prediction_heads.active_head_names == {"clf", "reg"}
    assert model_with_prediction_heads.active_prediction_head_names == {"clf"}


def test_model_enable_disable_unknown_head_raises(multitask_model: TorchkitModel):
    with pytest.raises(ValueError, match="does not exist"):
        multitask_model.enable_head("missing")

    with pytest.raises(ValueError, match="does not exist"):
        multitask_model.disable_head("missing")


# -------------------------
# Freeze / unfreeze
# -------------------------

def test_model_freeze_unfreeze_backbone(single_task_model: TorchkitModel):
    single_task_model.freeze_backbone()
    for p in single_task_model.backbone.parameters():
        assert p.requires_grad is False

    single_task_model.unfreeze_backbone()
    for p in single_task_model.backbone.parameters():
        assert p.requires_grad is True


def test_model_freeze_unfreeze_head(multitask_model: TorchkitModel):
    multitask_model.freeze_head("clf")
    for p in multitask_model.heads["clf"].parameters():
        assert p.requires_grad is False

    multitask_model.unfreeze_head("clf")
    for p in multitask_model.heads["clf"].parameters():
        assert p.requires_grad is True


def test_model_freeze_unfreeze_all_heads(multitask_model: TorchkitModel):
    multitask_model.freeze_all_heads()
    for head in multitask_model.heads.values():
        for p in head.parameters():
            assert p.requires_grad is False

    multitask_model.unfreeze_all_heads()
    for head in multitask_model.heads.values():
        for p in head.parameters():
            assert p.requires_grad is True


# -------------------------
# Forward routing
# -------------------------

def test_model_forward_accepts_tensor(single_task_model: TorchkitModel, x_tensor: Tensor):
    out = single_task_model(x_tensor)

    assert set(out.keys()) == {"clf"}
    assert "logits" in out["clf"]
    assert out["clf"]["logits"].shape == (4, 2)


def test_model_forward_accepts_payload_dict(single_task_model: TorchkitModel, payload: dict[str, Tensor]):
    out = single_task_model(payload)

    assert set(out.keys()) == {"clf"}
    assert "logits" in out["clf"]
    assert out["clf"]["logits"].shape == (4, 2)


def test_model_forward_requires_x_in_payload(single_task_model: TorchkitModel):
    with pytest.raises(KeyError, match="must contain key 'x'"):
        single_task_model({"tabular": torch.randn(4, 2)})


def test_model_forward_requires_x_tensor(single_task_model: TorchkitModel):
    with pytest.raises(TypeError, match="payload\\['x'\\] must be a Tensor"):
        single_task_model({"x": [1, 2, 3]})  # type: ignore[arg-type]


def test_model_forward_rejects_bad_input_type(single_task_model: TorchkitModel):
    with pytest.raises(TypeError, match="must be a Tensor or dict"):
        single_task_model([1, 2, 3])  # type: ignore[arg-type]


def test_model_forward_returns_requested_backbone_features_true(multitask_model: TorchkitModel, x_tensor: Tensor):
    out = multitask_model(x_tensor, return_backbone_features=True)

    assert "backbone" in out
    assert set(out["backbone"].keys()) == {"feat_a", "feat_b"}


def test_model_forward_returns_requested_backbone_feature_by_name(multitask_model: TorchkitModel, x_tensor: Tensor):
    out = multitask_model(x_tensor, return_backbone_features="feat_a")

    assert "backbone" in out
    assert set(out["backbone"].keys()) == {"feat_a"}


def test_model_forward_returns_requested_backbone_features_collection(multitask_model: TorchkitModel, x_tensor: Tensor):
    out = multitask_model(x_tensor, return_backbone_features=["feat_b"])

    assert "backbone" in out
    assert set(out["backbone"].keys()) == {"feat_b"}


def test_model_forward_only_runs_active_heads(multitask_model: TorchkitModel, x_tensor: Tensor):
    multitask_model.disable_head("reg")

    out = multitask_model(x_tensor)

    assert set(out.keys()) == {"clf"}


def test_model_forward_routes_payload_to_task_head(payload: dict[str, Tensor]):
    backbone = DummyBackbone()
    fuse = DummyFuse()
    adapter = AddOneAdapter()
    head_module = HeadWithPayload(in_features=6, out_features=2)

    head = TaskHead(
        required_features={"feat_a", "feat_b"},
        fuse_module=fuse,
        feature_adapter=adapter,
        head_module=head_module,
    )

    model = TorchkitModel(backbone=backbone, heads={"clf": head})
    _ = model(payload)

    assert fuse.last_payload is not None
    assert adapter.last_payload is not None
    assert head_module.last_payload is not None
    assert set(fuse.last_payload.keys()) == set(payload.keys())
    assert set(adapter.last_payload.keys()) == set(payload.keys())
    assert set(head_module.last_payload.keys()) == set(payload.keys())


def test_model_forward_raises_if_prediction_head_exists_but_logits_missing(x_tensor: Tensor):
    backbone = DummyBackbone()
    head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearPredictionsHead(in_features=3, out_features=1),
    )
    phead = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
    )

    model = TorchkitModel(backbone=backbone, heads={"reg": head}, prediction_heads={"reg": phead})

    with pytest.raises(KeyError, match="did not return 'logits'"):
        model(x_tensor)


# -------------------------
# Predict routing and restoration
# -------------------------

def test_model_predict_requires_task_names(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    with pytest.raises(ValueError, match="At least one task name must be specified"):
        model_with_prediction_heads.predict(x_tensor)


def test_model_predict_rejects_invalid_task_name(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    with pytest.raises(ValueError, match="All task names must exist"):
        model_with_prediction_heads.predict(x_tensor, "missing")


def test_model_predict_single_task_without_raw_head_outputs(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    out = model_with_prediction_heads.predict(x_tensor, "clf", return_raw_head_outputs=False)

    assert set(out.keys()) == {"clf"}
    assert "logits" in out["clf"]
    assert "calibrated_logits" in out["clf"]
    assert "probabilities" in out["clf"]
    assert "predictions" in out["clf"]

def test_model_predict_single_task_with_raw_head_outputs(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    out = model_with_prediction_heads.predict(x_tensor, "clf", return_raw_head_outputs=True)

    assert set(out.keys()) == {"clf"}
    assert "logits" in out["clf"]
    assert "calibrated_logits" in out["clf"]
    assert "probabilities" in out["clf"]
    assert "predictions" in out["clf"]


def test_model_predict_multitask_mixed_prediction_heads(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    out = model_with_prediction_heads.predict(x_tensor, "clf", "reg", return_raw_head_outputs=True)

    assert set(out.keys()) == {"clf", "reg"}

    # clf has prediction head => enriched
    assert "logits" in out["clf"]
    assert "calibrated_logits" in out["clf"]
    assert "probabilities" in out["clf"]
    assert "predictions" in out["clf"]

    # reg has no prediction head => raw output only
    assert "predictions" in out["reg"]
    assert "calibrated_logits" not in out["reg"]
    assert "probabilities" not in out["reg"]


def test_model_predict_can_return_backbone_features(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    out = model_with_prediction_heads.predict(
        x_tensor,
        "clf",
        return_backbone_features=True,
        return_raw_head_outputs=True,
    )

    assert "backbone" in out
    assert "clf" in out
    assert set(out["backbone"].keys()) == {"feat_a"}


def test_model_predict_restores_previous_active_heads(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    model_with_prediction_heads.disable_head("reg")
    previous_active = copy.deepcopy(model_with_prediction_heads.active_head_names)

    _ = model_with_prediction_heads.predict(x_tensor, "clf", return_raw_head_outputs=True)

    assert model_with_prediction_heads.active_head_names == previous_active


def test_model_predict_temporarily_overrides_inactive_requested_head(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    model_with_prediction_heads.disable_head("clf")
    assert "clf" not in model_with_prediction_heads.active_head_names

    out = model_with_prediction_heads.predict(x_tensor, "clf", return_raw_head_outputs=True)

    assert "clf" in out
    # restore after call
    assert "clf" not in model_with_prediction_heads.active_head_names


def test_model_predict_without_prediction_head_returns_raw_output(multitask_model: TorchkitModel, x_tensor: Tensor):
    out = multitask_model.predict(x_tensor, "reg")

    assert set(out.keys()) == {"reg"}
    assert "predictions" in out["reg"]
    assert "logits" not in out["reg"]


def test_model_predict_disabled_calibrator_removes_calibrated_logits(model_with_prediction_heads: TorchkitModel, x_tensor: Tensor):
    out_enabled = model_with_prediction_heads.predict(x_tensor, "clf", return_raw_head_outputs=True)
    assert "calibrated_logits" in out_enabled["clf"]

    model_with_prediction_heads.prediction_heads["clf"].calibrator.disable()

    out_disabled = model_with_prediction_heads.predict(x_tensor, "clf", return_raw_head_outputs=True)
    assert "logits" in out_disabled["clf"]
    assert "calibrated_logits" not in out_disabled["clf"]
    assert "probabilities" in out_disabled["clf"]
    assert "predictions" in out_disabled["clf"]


# -------------------------
# Store / load helpers
# -------------------------

def test_model_store_and_load_roundtrip(single_task_model: TorchkitModel, x_tensor: Tensor, tmp_path):
    path = tmp_path / "model.pt"

    original_state = copy.deepcopy(single_task_model.state_dict())
    single_task_model.store(str(path))

    for p in single_task_model.parameters():
        with torch.no_grad():
            p.add_(1.0)

    single_task_model.load(str(path))

    for k, v in original_state.items():
        assert torch.allclose(v, single_task_model.state_dict()[k])


def test_model_validate_load_path_rejects_bad_path():
    with pytest.raises(TypeError):
        TorchkitModel.validate_load_path(123)  # type: ignore[arg-type]

    with pytest.raises(FileNotFoundError):
        TorchkitModel.validate_load_path("definitely_missing_file.pt")


# -------------------------
# Factory
# -------------------------

def _make_model_spec() -> TorchkitModelSpec:
    return TorchkitModelSpec(
        backbone=BackboneSpec(
            cls=DummyBackbone,
            kwargs={},
        ),
        heads={
            "clf": TaskHeadSpec(
                required_features="feat_a",
                feature_adapter=FeatureAdapterSpec(cls=IdentityAdapter, kwargs={}),
                head_module=HeadModuleSpec(cls=LinearLogitsHead, kwargs={"in_features": 3, "out_features": 2}),
                active=True,
            ),
            "reg": TaskHeadSpec(
                required_features={"feat_a", "feat_b"},
                fuse_module=FuseModuleSpec(cls=DummyFuse, kwargs={}),
                feature_adapter=FeatureAdapterSpec(cls=IdentityAdapter, kwargs={}),
                head_module=HeadModuleSpec(cls=LinearPredictionsHead, kwargs={"in_features": 6, "out_features": 1}),
                active=True,
            ),
        },
        prediction_heads={
            "clf": PredictionHeadSpec(
                calibrator=CalibratorSpec(cls=DummyCalibrator, kwargs={}, active=True),
                probability_mapper=ProbabilityMapperSpec(cls=DummyProbabilityMapper, kwargs={}),
                decision_module=DecisionModuleSpec(cls=DummyDecisionModule, kwargs={}),
                active=True,
            )
        },
    )


def test_model_factory_builds_model():
    spec = _make_model_spec()

    model = TorchkitModelFactory.build(spec)

    assert isinstance(model, TorchkitModel)
    assert model.head_names == {"clf", "reg"}
    assert model.prediction_head_names == {"clf"}


def test_model_factory_rejects_missing_backbone():
    spec = TorchkitModelSpec(backbone=None, heads={"clf": TaskHeadSpec(required_features="feat_a")})

    with pytest.raises(ValueError, match="backbone must be specified"):
        TorchkitModelFactory.build(spec)


def test_model_factory_rejects_empty_heads():
    spec = TorchkitModelSpec(
        backbone=BackboneSpec(cls=DummyBackbone, kwargs={}),
        heads={},
    )

    with pytest.raises(ValueError, match="heads must be a non-empty dict"):
        TorchkitModelFactory.build(spec)


def test_model_factory_rejects_prediction_head_extra_key():
    spec = _make_model_spec()
    spec.prediction_heads["other"] = copy.deepcopy(spec.prediction_heads["clf"])

    with pytest.raises(ValueError, match="contains keys not present in heads"):
        TorchkitModelFactory.build(spec)


def test_model_factory_rejects_whole_and_nested_loading_mix():
    spec = _make_model_spec()

    with pytest.raises(ValueError, match="cannot be mixed with nested component state loading"):
        TorchkitModelFactory.build(
            spec,
            state_dict={},
            backbone_state_dict={},
        )


def test_model_factory_can_load_whole_state_dict(x_tensor: Tensor):
    spec = _make_model_spec()
    original = TorchkitModelFactory.build(spec)
    state_dict = original.state_dict()

    loaded = TorchkitModelFactory.build(spec, state_dict=state_dict)

    out_orig = original.predict(x_tensor, "clf", "reg", return_raw_head_outputs=True)
    out_loaded = loaded.predict(x_tensor, "clf", "reg", return_raw_head_outputs=True)

    for k in state_dict:
        assert torch.allclose(state_dict[k], loaded.state_dict()[k])

    assert out_orig["clf"]["logits"].shape == out_loaded["clf"]["logits"].shape
    assert out_orig["reg"]["predictions"].shape == out_loaded["reg"]["predictions"].shape


def test_model_factory_can_load_nested_component_state_dicts():
    spec = _make_model_spec()
    original = TorchkitModelFactory.build(spec)

    backbone_sd = copy.deepcopy(original.backbone.state_dict())
    clf_head_sd = copy.deepcopy(original.heads["clf"].state_dict())
    reg_head_sd = copy.deepcopy(original.heads["reg"].state_dict())
    clf_phead_sd = copy.deepcopy(original.prediction_heads["clf"].state_dict())

    loaded = TorchkitModelFactory.build(
        spec,
        backbone_state_dict=backbone_sd,
        head_state_dicts={"clf": clf_head_sd, "reg": reg_head_sd},
        prediction_head_state_dicts={"clf": clf_phead_sd},
    )

    for k, v in backbone_sd.items():
        assert torch.allclose(v, loaded.backbone.state_dict()[k])

    for k, v in clf_head_sd.items():
        assert torch.allclose(v, loaded.heads["clf"].state_dict()[k])

    for k, v in reg_head_sd.items():
        assert torch.allclose(v, loaded.heads["reg"].state_dict()[k])

    for k, v in clf_phead_sd.items():
        assert torch.allclose(v, loaded.prediction_heads["clf"].state_dict()[k])


def test_model_factory_can_load_nested_head_component_state_dicts():
    spec = _make_model_spec()
    original = TorchkitModelFactory.build(spec)

    reg_fuse_sd = copy.deepcopy(original.heads["reg"].fuse_module.state_dict())
    clf_adapter_sd = copy.deepcopy(original.heads["clf"].feature_adapter.state_dict())
    clf_head_module_sd = copy.deepcopy(original.heads["clf"].head_module.state_dict())

    loaded = TorchkitModelFactory.build(
        spec,
        head_component_state_dicts={
            "reg": {"fuse_module": reg_fuse_sd},
            "clf": {
                "feature_adapter": clf_adapter_sd,
                "head_module": clf_head_module_sd,
            },
        },
    )

    for k, v in reg_fuse_sd.items():
        assert torch.allclose(v, loaded.heads["reg"].fuse_module.state_dict()[k])

    for k, v in clf_adapter_sd.items():
        assert torch.allclose(v, loaded.heads["clf"].feature_adapter.state_dict()[k])

    for k, v in clf_head_module_sd.items():
        assert torch.allclose(v, loaded.heads["clf"].head_module.state_dict()[k])