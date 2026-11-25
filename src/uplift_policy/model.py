import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_STATE = 42


class TLearner:
    """Separate arm models; uplift = mu1(x) - mu0(x)."""

    def __init__(self, base_learner=None):
        self.base_learner = (
            base_learner if base_learner is not None
            else HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        )

    def fit(self, X, treatment, outcome):
        X = np.asarray(X)
        t = np.asarray(treatment).astype(int)
        y = np.asarray(outcome).astype(int)
        self.model_control_ = clone(self.base_learner).fit(X[t == 0], y[t == 0])
        self.model_treated_ = clone(self.base_learner).fit(X[t == 1], y[t == 1])
        return self

    def predict_uplift(self, X):
        X = np.asarray(X)
        return self.model_treated_.predict_proba(X)[:, 1] - self.model_control_.predict_proba(X)[:, 1]
