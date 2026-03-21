from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from numbers import Number
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.models.Model._model import TorchkitModel


def _move_to_device(x: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x


def _infer_batch_size(batch_dict: dict[str, Any]) -> int:
    x = batch_dict.get("x", None)
    if x is None:
        raise KeyError(
            "Expected batch to contain a Tensor 'x' key for primary model input, but it was not found. "
            "Cannot infer batch size."
        )
    if not torch.is_tensor(x):
        raise TypeError(
            f"'x' is supposed to be a Tensor for primary model input, but got {type(x).__name__}. "
            "Cannot infer batch size."
        )
    if x.ndim == 0:
        raise ValueError("batch['x'] is scalar; cannot infer batch size.")
    return int(x.shape[0])


def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
    cur = root
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError(f"Invalid empty path: {path!r}")
    for part in parts[:-1]:
        nxt = cur.get(part, None)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, dict):
            raise TypeError(f"Cannot set into non-dict at path segment {part!r} for full path {path!r}.")
        cur = nxt
    cur[parts[-1]] = value


def _append_cached(cache: dict[str, list[torch.Tensor]], key: str, tensor: torch.Tensor) -> None:
    cache[key].append(tensor.detach().cpu())


def _cat_cached_list(ts: list[torch.Tensor], key: str) -> torch.Tensor:
    if len(ts) == 0:
        raise ValueError(f"Empty cache list for key {key!r} (unexpected).")
    if len(ts) == 1:
        return ts[0]
    if ts[0].ndim == 0:
        return torch.stack(ts, dim=0)
    return torch.cat(ts, dim=0)


def _as_report_leaf(value: Any) -> tuple[Any, bool]:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item()), True
        return value.detach().cpu().tolist(), False

    if isinstance(value, bool):
        return value, False

    if isinstance(value, Number):
        return float(value), True

    return deepcopy(value), False


def _merge_report_dicts(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key in dst:
            raise KeyError(
                f"Report metric key collision for {key!r}. "
                "Use unique metric names or a CompositeReportEvaluator with distinct child names."
            )
        dst[key] = deepcopy(value)


def evaluate(
    model: TorchkitModel,
    data: DataLoader | Dataset,
    evaluator: BundleReportEvaluator,
    *,
    device: Optional[torch.device | str] = None,
    backbone_kwargs: Optional[dict[str, Any]] = None,
    head_kwargs: Optional[dict[str, dict[str, Any]]] = None,
    use_amp: bool = False,
    dataloader_factory: Optional[Callable[[Dataset], DataLoader]] = None,
) -> dict[str, Any]:
    """
    Evaluate a model with a report-evaluator bundle on a dataloader or dataset.

    Batch-level report metrics are aggregated across batches as follows:
    - numeric leaves are weighted by batch size and averaged over the full loader
    - non-numeric leaves are collected into ordered per-batch lists

    Dataset-level report metrics are computed once from epoch-wise cached tensors.
    The returned dictionary is the merged union of both evaluator outputs.
    """
    if not isinstance(model, TorchkitModel):
        raise TypeError(f"`model` must be a TorchkitModel, got {type(model).__name__}.")
    if not isinstance(evaluator, BundleReportEvaluator):
        raise TypeError(
            f"`evaluator` must be a BundleReportEvaluator, got {type(evaluator).__name__}."
        )

    if isinstance(data, DataLoader):
        loader = data
    elif isinstance(data, Dataset):
        if dataloader_factory is None:
            loader = DataLoader(data, batch_size=1, shuffle=False)
        else:
            loader = dataloader_factory(data)
            if not isinstance(loader, DataLoader):
                raise TypeError(
                    "dataloader_factory must return a torch.utils.data.DataLoader, "
                    f"got {type(loader).__name__}."
                )
    else:
        raise TypeError(
            f"`data` must be a DataLoader or Dataset, got {type(data).__name__}."
        )

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            try:
                device = next(model.buffers()).device
            except StopIteration:
                device = torch.device("cpu")
    if not isinstance(device, torch.device):
        device = torch.device(device)

    batch_evaluator = evaluator.batch_evaluator
    dataset_evaluator = evaluator.dataset_evaluator
    dataset_required_keys = tuple(evaluator.dataset_required_keys)
    dataset_optional_keys = tuple(evaluator.dataset_optional_keys)
    dataset_optional_key_set = set(dataset_optional_keys)

    dataset_cache: dict[str, list[torch.Tensor]] = defaultdict(list)
    dataset_none_keys: set[str] = set()

    batch_numeric_weight_sums: dict[str, float] = {}
    batch_numeric_total_weights: dict[str, float] = {}
    batch_sequence_values: dict[str, list[Any]] = defaultdict(list)

    was_training = model.training
    num_batches = 0
    model.eval()

    try:
        with torch.no_grad():
            for batch in loader:
                if not isinstance(batch, dict):
                    raise TypeError(
                        f"Expected batch as dict[str, Any], got {type(batch)}. "
                        "Use a collate_fn that returns a dict. Names are required for routing."
                    )
                if "x" not in batch:
                    raise KeyError(
                        f"Expected batch to contain the 'x' key for the primary model input, but got keys: {list(batch.keys())}. "
                        "Use a collate_fn that puts model inputs under 'x'. This is an enforced convention for this library."
                    )

                num_batches += 1
                batch_size = _infer_batch_size(batch)
                batch = _move_to_device(batch, device)

                if use_amp:
                    with torch.autocast(device_type=device.type):
                        model_out = model.predict(
                            batch,
                            *model.active_head_names,
                            backbone_kwargs=backbone_kwargs,
                            head_kwargs=head_kwargs,
                            return_raw_head_outputs=True,
                        )
                else:
                    model_out = model.predict(
                        batch,
                        *model.active_head_names,
                        backbone_kwargs=backbone_kwargs,
                        head_kwargs=head_kwargs,
                        return_raw_head_outputs=True,
                    )

                eval_in = dict(model_out)
                eval_in["batch"] = batch

                if batch_evaluator is not None:
                    batch_metrics = batch_evaluator(inputs=eval_in)
                    for key, value in batch_metrics.items():
                        normalized_value, is_numeric = _as_report_leaf(value)
                        if is_numeric:
                            batch_numeric_weight_sums[key] = (
                                batch_numeric_weight_sums.get(key, 0.0)
                                + float(normalized_value) * float(batch_size)
                            )
                            batch_numeric_total_weights[key] = (
                                batch_numeric_total_weights.get(key, 0.0) + float(batch_size)
                            )
                        else:
                            batch_sequence_values[key].append(normalized_value)

                if dataset_evaluator is not None:
                    for key in dataset_required_keys + dataset_optional_keys:
                        is_optional = key in dataset_optional_key_set
                        value = dataset_evaluator.resolve(eval_in, key, strict=not is_optional)
                        if value is None:
                            if key in dataset_cache:
                                raise ValueError(
                                    f"Dataset report key {key!r} resolved to both Tensor values and None across batches."
                                )
                            dataset_none_keys.add(key)
                            continue
                        if key in dataset_none_keys:
                            raise ValueError(
                                f"Dataset report key {key!r} resolved to both None and Tensor values across batches."
                            )
                        _append_cached(dataset_cache, key, value)
    finally:
        if was_training:
            model.train()

    if num_batches == 0:
        raise ValueError("Evaluation loader produced 0 batches.")

    out: dict[str, Any] = {}

    if batch_evaluator is not None:
        batch_results: dict[str, Any] = {}
        for key, weighted_sum in batch_numeric_weight_sums.items():
            total_weight = batch_numeric_total_weights[key]
            batch_results[key] = weighted_sum / total_weight
        for key, values in batch_sequence_values.items():
            batch_results[key] = deepcopy(values)
        _merge_report_dicts(out, batch_results)

    if dataset_evaluator is not None:
        dataset_inputs: dict[str, Any] = {}
        for key, tensors in dataset_cache.items():
            _set_by_path(dataset_inputs, key, _cat_cached_list(tensors, key=key))
        for key in dataset_none_keys:
            _set_by_path(dataset_inputs, key, None)
        dataset_results = dataset_evaluator(inputs=dataset_inputs)
        _merge_report_dicts(out, dataset_results)

    return out
