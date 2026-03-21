from __future__ import annotations

import pytest
import torch

from torchkit.evaluate import evaluate
from torchkit.evaluate.report._report_evaluator import ReportEvaluator
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelFactory
from torchkit.models.Model.factory import TorchkitModelSpec
from torchkit.models.adapters._feature_adapter import IdentityAdapter
from torchkit.models.adapters.factory import FeatureAdapterSpec
from torchkit.models.backbone._backbone import Backbone
from torchkit.models.backbone.factory import BackboneSpec
from torchkit.models.head.factory import TaskHeadSpec
from torchkit.models.head_module.factory import HeadModuleSpec
from torchkit.models.prediction.factory import PredictionHeadSpec
from torchkit.data._dataset import TorchkitDataset


class TinyEvalDataset(TorchkitDataset):
    def __init__(self) -> None:
        self._xs = []
        self._ys = []
        for _ in range(8):
            self._xs.append(torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32))
            self._ys.append(0)
            self._xs.append(torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32))
            self._ys.append(1)

    def __len__(self) -> int:
        return len(self._xs)

    def my_getitem(self, index):
        return {
            "x": self._xs[index].clone(),
            "y": torch.tensor(self._ys[index], dtype=torch.long),
        }


class DeterministicBackbone(Backbone):
    def __init__(self, scale_factor: float = 1.0):
        super().__init__(supported_features={"features"})
        self.scale_factor = float(scale_factor)
        self._dummy = torch.nn.Parameter(torch.tensor(0.0))

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        x = input["x"]
        return {"features": self.scale_factor * x + (0.0 * self._dummy)}


class DirectBinaryLogitsHead(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": torch.stack([x[:, 0], x[:, 1]], dim=1)}


class PositiveLogitMeanReportEvaluator(ReportEvaluator):
    def __init__(self, *, score_key: str, name: str = "batch_logits") -> None:
        super().__init__(name=name)
        self.score_key = score_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.score_key,)

    def metrics(self, *, inputs: dict[str, object]) -> dict[str, object]:
        logits = self.resolve(inputs, self.score_key).detach()
        return {
            "positive_logit_mean": float(logits[:, 1].float().mean().item()),
            "batch_pred_labels": torch.argmax(logits, dim=1).detach().cpu().tolist(),
        }


class AccuracyReportEvaluator(ReportEvaluator):
    def __init__(self, *, score_key: str, target_key: str, name: str = "clf") -> None:
        super().__init__(name=name)
        self.score_key = score_key
        self.target_key = target_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.score_key, self.target_key)

    def metrics(self, *, inputs: dict[str, object]) -> dict[str, object]:
        logits = self.resolve(inputs, self.score_key).detach()
        targets = self.resolve(inputs, self.target_key).detach()
        preds = torch.argmax(logits, dim=1)
        return {
            "accuracy": float((preds == targets).float().mean().item()),
            "n_samples": int(targets.shape[0]),
        }


def _make_model() -> TorchkitModel:
    spec = TorchkitModelSpec(
        backbone=BackboneSpec(
            cls=DeterministicBackbone,
            kwargs={"scale_factor": 1.0},
        ),
        heads={
            "clf": TaskHeadSpec(
                required_features="features",
                feature_adapter=FeatureAdapterSpec(cls=IdentityAdapter, kwargs={}),
                head_module=HeadModuleSpec(cls=DirectBinaryLogitsHead, kwargs={}),
                active=True,
            )
        },
        prediction_heads={
            "clf": PredictionHeadSpec(
                probability_mapper=None,
                decision_module=None,
                active=True,
            )
        },
    )
    return TorchkitModelFactory.build(spec, device="cpu")


def test_evaluate_runs_bundle_on_dataloader_and_aggregates_batch_reports() -> None:
    dataset = TinyEvalDataset()
    model = _make_model()
    loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)

    results = evaluate(
        model,
        loader,
        BundleReportEvaluator(
            batch_evaluator=PositiveLogitMeanReportEvaluator(score_key="clf/logits"),
            dataset_evaluator=AccuracyReportEvaluator(
                score_key="clf/logits",
                target_key="batch/y",
                name="clf",
            ),
        ),
        device="cpu",
    )

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["n_samples"] == len(dataset)
    assert results["positive_logit_mean"] == pytest.approx(1.0)

    batch_label_lists = results["batch_pred_labels"]
    assert isinstance(batch_label_lists, list)
    assert [len(x) for x in batch_label_lists] == [3, 3, 3, 3, 3, 1]
    assert batch_label_lists[0] == [0, 1, 0]
    assert batch_label_lists[-1] == [1]


def test_evaluate_accepts_dataset_with_custom_dataloader_factory() -> None:
    dataset = TinyEvalDataset()
    model = _make_model()

    results = evaluate(
        model,
        dataset,
        BundleReportEvaluator(
            dataset_evaluator=AccuracyReportEvaluator(
                score_key="clf/logits",
                target_key="batch/y",
                name="clf",
            ),
        ),
        device="cpu",
        dataloader_factory=lambda ds: torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False),
    )

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["n_samples"] == len(dataset)
