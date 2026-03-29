from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from torchkit.models.Model._model import TorchkitModel
from torchkit.models.backbone._backbone import Backbone
from torchkit.models.head._task_head import TaskHead

from torchkit.models.fuse._fuse_module import FuseModule
from torchkit.models.adapters._feature_adapter import FeatureAdapter

from torchkit.objectives.relational import CELoss, MSELoss, BCELoss
from torchkit.objectives.Multitask import MultitaskObjective


# ============================================================
# Dummy model components
# ============================================================

class DummyBackbone(Backbone):
    def __init__(self):
        super().__init__(supported_features=["feat_a", "feat_b"])

    def _forward_impl(
        self,
        input: dict[str, Tensor],
        *,
        requested_features=None,
        **kwargs,
    ) -> dict[str, Tensor]:
        x = input["x"]
        out = {}
        if "feat_a" in requested_features:
            out["feat_a"] = x + 1.0
        if "feat_b" in requested_features:
            out["feat_b"] = x + 2.0
        return out


class DummyFuse(FuseModule):
    def forward(self, features: dict[str, Tensor], **kwargs) -> Tensor:
        return torch.cat([features[k] for k in sorted(features.keys())], dim=1)


class IdentityAdapter(FeatureAdapter):
    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features


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


class LinearProbabilitiesHead(nn.Module):
    """
    For BCE integration without involving prediction heads.
    """
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        logits = self.linear(x)
        return {"probabilities": torch.sigmoid(logits)}


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def x_tensor() -> Tensor:
    return torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.1, 1.2, -0.4],
            [-0.3, 0.7, 1.5],
            [1.1, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def batch_payload(x_tensor: Tensor) -> dict[str, Tensor]:
    return {
        "x": x_tensor.clone(),
        "y_clf": torch.tensor([0, 1, 1, 0], dtype=torch.long),
        "y_reg": torch.tensor([[1.5], [2.0], [2.5], [1.0]], dtype=torch.float32),
        "y_bce": torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32),
    }


@pytest.fixture
def classification_model() -> TorchkitModel:
    backbone = DummyBackbone()
    clf_head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )
    model = TorchkitModel(backbone=backbone, heads={"clf": clf_head})
    model.train()
    return model


@pytest.fixture
def regression_model() -> TorchkitModel:
    backbone = DummyBackbone()
    reg_head = TaskHead(
        required_features={"feat_a", "feat_b"},
        fuse_module=DummyFuse(),
        feature_adapter=IdentityAdapter(),
        head_module=LinearPredictionsHead(in_features=6, out_features=1),
    )
    model = TorchkitModel(backbone=backbone, heads={"reg": reg_head})
    model.train()
    return model


@pytest.fixture
def bce_model() -> TorchkitModel:
    backbone = DummyBackbone()
    clf_head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearProbabilitiesHead(in_features=3, out_features=1),
    )
    model = TorchkitModel(backbone=backbone, heads={"clf": clf_head})
    model.train()
    return model


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

    model = TorchkitModel(
        backbone=backbone,
        heads={"clf": clf_head, "reg": reg_head},
    )
    model.train()
    return model


# ============================================================
# Integration tests: model.forward -> objective -> scalar loss
# ============================================================

def test_model_forward_outputs_work_with_ce_objective(
    classification_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = classification_model(batch_payload)

    objective_inputs = dict(model_out)
    objective_inputs["batch"] = {"y": batch_payload["y_clf"]}

    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    loss = objective(inputs=objective_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_forward_outputs_work_with_mse_objective(
    regression_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = regression_model(batch_payload)

    objective_inputs = dict(model_out)
    objective_inputs["batch"] = {"target": batch_payload["y_reg"]}

    objective = MSELoss(
        input_path="reg/predictions",
        target_path="batch/target",
        reduction="mean",
    )

    loss = objective(inputs=objective_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_forward_outputs_work_with_bce_objective(
    bce_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = bce_model(batch_payload)

    objective_inputs = dict(model_out)
    objective_inputs["batch"] = {"target": batch_payload["y_bce"]}

    objective = BCELoss(
        input_path="clf/probabilities",
        target_path="batch/target",
        reduction="mean",
    )

    loss = objective(inputs=objective_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_forward_outputs_work_with_multitask_objective(
    multitask_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = multitask_model(batch_payload)

    objective_inputs = dict(model_out)
    objective_inputs["batch"] = {
        "y": batch_payload["y_clf"],
        "target": batch_payload["y_reg"],
    }

    ce = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
        weight=1.0,
    )
    mse = MSELoss(
        input_path="reg/predictions",
        target_path="batch/target",
        reduction="mean",
        weight=0.5,
    )

    objective = MultitaskObjective(
        {
            "clf": ce,
            "reg": mse,
        },
        name="multi",
    )

    loss = objective(inputs=objective_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "clf" in objective.per_objective_loss
    assert "reg" in objective.per_objective_loss


def test_ce_objective_loss_matches_manual_forward_application(
    classification_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = classification_model(batch_payload)
    logits = model_out["clf"]["logits"]
    target = batch_payload["y_clf"]

    objective_inputs = {
        "clf": {"logits": logits},
        "batch": {"y": target},
    }

    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    loss = objective(inputs=objective_inputs)
    expected = torch.nn.functional.cross_entropy(logits, target, reduction="mean")

    assert torch.allclose(loss, expected)


def test_mse_objective_loss_matches_manual_forward_application(
    regression_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = regression_model(batch_payload)
    preds = model_out["reg"]["predictions"]
    target = batch_payload["y_reg"]

    objective_inputs = {
        "reg": {"predictions": preds},
        "batch": {"target": target},
    }

    objective = MSELoss(
        input_path="reg/predictions",
        target_path="batch/target",
        reduction="mean",
    )

    loss = objective(inputs=objective_inputs)
    expected = torch.nn.functional.mse_loss(preds, target, reduction="mean")

    assert torch.allclose(loss, expected)


def test_loss_from_model_forward_supports_backward(
    multitask_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = multitask_model(batch_payload)

    objective_inputs = dict(model_out)
    objective_inputs["batch"] = {
        "y": batch_payload["y_clf"],
        "target": batch_payload["y_reg"],
    }

    objective = MultitaskObjective(
        {
            "clf": CELoss(
                input_path="clf/logits",
                target_path="batch/y",
                reduction="mean",
            ),
            "reg": MSELoss(
                input_path="reg/predictions",
                target_path="batch/target",
                reduction="mean",
            ),
        },
        name="multi",
    )

    loss = objective(inputs=objective_inputs)
    loss.backward()

    has_grad = any(
        param.grad is not None
        for param in multitask_model.parameters()
        if param.requires_grad
    )
    assert has_grad


def test_optional_objective_can_zero_out_on_missing_branch(
    regression_model: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    model_out = regression_model(batch_payload)

    objective_inputs = dict(model_out)
    objective_inputs["batch"] = {"target": batch_payload["y_reg"]}

    # Missing clf/logits on purpose, but optional=True
    optional_ce = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
        is_optional=True,
    )

    required_mse = MSELoss(
        input_path="reg/predictions",
        target_path="batch/target",
        reduction="mean",
        is_optional=False,
    )

    objective = MultitaskObjective(
        {
            "clf": optional_ce,
            "reg": required_mse,
        },
        name="multi_optional",
    )

    loss = objective(inputs=objective_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
