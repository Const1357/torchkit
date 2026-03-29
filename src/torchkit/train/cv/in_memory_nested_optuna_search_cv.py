from __future__ import annotations

import copy
import os

from torchkit.train.cv.in_memory_optuna_search_cv import InMemoryOptunaSearchCV
from torchkit.train.cv.nested_optuna_search_cv import NestedOptunaSearchCV


class InMemoryNestedOptunaSearchCV(NestedOptunaSearchCV):
    """
    Nested CV variant that uses ``InMemoryOptunaSearchCV`` for each inner search.
    """

    def _build_inner_search(self, *, outer_fold: int) -> InMemoryOptunaSearchCV:
        return InMemoryOptunaSearchCV(
            model_spec=copy.deepcopy(self.model_spec),
            trainer_spec=copy.deepcopy(self.trainer_spec),
            parameter_grid=copy.deepcopy(self.parameter_grid),
            splitter_cls=self.inner_splitter_cls,
            dataloader_factory=self.dataloader_factory,
            n_trials=self.n_trials,
            max_trial_attempts=self.max_trial_attempts,
            n_splits=self.k_inner if self.k_inner is not None else 0,
            shuffle=self.shuffle_inner,
            random_state=self.random_state,
            calibrate=self.calibrate,
            report_evaluator=copy.deepcopy(self.report_evaluator),
            logging=self.logging,
            _log_root_dir=(
                os.path.join(self.log_dir, "outer_folds", f"outer_fold_{outer_fold:03d}", "inner_search")
                if self.logging and self.log_dir is not None
                else None
            ),
            final_model_dir=self._outer_fold_model_dir(outer_fold),
            keep_final_model_state_dict_cpu=self.keep_final_model_state_dict_cpu,
        )
