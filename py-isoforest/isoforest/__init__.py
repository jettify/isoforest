from typing import Optional, Union

import numpy as np
from isoforest._isoforest import PyIsoForest

__all__ = ["IsolationForest"]
__version__ = '0.0.1'


class IsolationForest(object):
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples="auto",
        contamination="auto",
        max_features: Union[int, float] = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.seed = seed or 0
        self._fited = True
        self._model = PyIsoForest(self.n_estimators, self.max_features, self.max_samples, seed=self.seed)

    def fit(self, X: np.ndarray):
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(X)
        return preds

    def decision_function(self, X) -> np.ndarray:
        preds = self._model.decision_function(X)
        return preds

    def score_samples(self, X) -> np.ndarray:
        return -self.decision_function(X)
