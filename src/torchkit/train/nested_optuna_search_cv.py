from __future__ import annotations
# placeholder

# TODO:
# For each outer fold:
    # Create Optuna study.
    # For each trial:
        # suggest hyperparameters
        # build new model + new trainer for that trial
        # run inner KFold
            # between inner folds, use trainer.reset_state() only because architecture is fixed within the trial
        # aggregate inner-fold metric(s)
        # return that aggregate to Optuna

        # Take best hyperparameters.
        # Build a fresh final model + trainer with those hyperparameters.
        # Retrain on full outer-train.

        # Fit calibrator from outer-train OOF logits. (final model for this fold is ready)

        # Evaluate on outer-test.
        # Store fold results.
        # Aggregate across outer folds for final reporting.

# NOTE: We should also store the best hyperparameters for each outer fold
# and the corresponding inner-fold metrics (aggregated)
# to analyze the stability of the hyperparameter selection across folds.

import optuna

from torchkit.data.split import KFoldSplitter
from torchkit.train.trainer import Trainer
from typing import Optional


class NestedOptunaSearchCV:

    def __init__(
        self,
        trainer: Trainer,
        parameter_grid: dict[str, list],    # resolve? trainer/X or just X?
        
        outer_splitter_cls: type[KFoldSplitter],
        inner_splitter_cls: type[KFoldSplitter],
        k_outer: int = 5,
        k_inner: int = 3,
        shuffle_outer: bool = False,
        shuffle_inner: bool = False,
        random_state: Optional[int] = None,

        calibrate: bool = True,    # whether to fit a calibrator on the OOF logits from the outer-train folds

    ):
        self.trainer = trainer

        self.outer_splitter = outer_splitter_cls(
            n_splits=k_outer,
            shuffle=shuffle_outer,
            random_state=random_state,
        )

        self.inner_splitter = inner_splitter_cls(
            n_splits=k_inner,
            shuffle=shuffle_inner,
            random_state=random_state,
        )

    # add bookkeeping here, best params etc. 
    # define and initialize the optuna study
    

    @staticmethod
    def infer_suggestion_type():
        pass

    @staticmethod
    def suggest_parameters(trial: optuna.Trial, parameter_grid: dict[str, list]) -> dict[str, any]:
        suggested_params = {}
        for param_name, param_values in parameter_grid.items():
            suggested_params[param_name] = trial.suggest_categorical(param_name, param_values)
        return suggested_params