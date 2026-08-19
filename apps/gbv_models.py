"""Version-independent inference for the trained GBV-form classifiers.

Replaces `joblib.load('models.joblib')`. The pickle was written by
scikit-learn 0.23.1 and its behaviour changes with the installed version (see
`tools/export_model_params.py`); these classes evaluate the learned parameters
directly, so predictions depend only on numpy.

Both estimators expose the scikit-learn surface the app uses -- `predict` and
`predict_proba` -- so they are drop-in replacements.

One deliberate behaviour change
-------------------------------
`LogisticRegressionOvR.predict_proba` uses the one-vs-rest scheme the model was
trained with (per-class sigmoid, then normalise). Modern scikit-learn applies
softmax to the same coefficients, which is a different scheme and is
substantially overconfident here. `predict` is unaffected: it is the argmax of
the decision function under either scheme.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models.npz"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # split form avoids overflow for large |z|
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


class LogisticRegressionOvR:
    """Multinomial-free logistic regression, scored one-vs-rest as trained."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, classes: np.ndarray):
        self.coef_ = coef
        self.intercept_ = intercept
        self.classes_ = classes

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        total = p.sum(axis=1, keepdims=True)
        # a row of all-zero sigmoids cannot happen for finite scores, but guard
        # against division by zero rather than emit nan
        np.divide(p, np.where(total == 0, 1.0, total), out=p)
        return p


class LinearSVCOvO:
    """Linear SVC scored one-vs-one, with libsvm's Platt/pairwise-coupling probabilities.

    Reproduces libsvm's `svm_predict_probability`: a Platt sigmoid per class
    pair, then the Wu-Lin-Weng coupling that turns pairwise probabilities into a
    distribution over classes.
    """

    _MIN_P = 1e-7

    def __init__(self, coef: np.ndarray, intercept: np.ndarray,
                 probA: np.ndarray, probB: np.ndarray, classes: np.ndarray):
        self.coef_ = coef
        self.intercept_ = intercept
        self.probA_ = probA
        self.probB_ = probB
        self.classes_ = classes
        self._pairs = list(itertools.combinations(range(len(classes)), 2))

    def _pairwise_decision(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_.T + self.intercept_

    def predict(self, X) -> np.ndarray:
        dec = self._pairwise_decision(X)
        votes = np.zeros((dec.shape[0], len(self.classes_)), dtype=np.int32)
        for k, (i, j) in enumerate(self._pairs):
            wins_i = dec[:, k] > 0
            votes[wins_i, i] += 1
            votes[~wins_i, j] += 1
        return self.classes_[np.argmax(votes, axis=1)]

    def _platt(self, d: np.ndarray, a: float, b: float) -> np.ndarray:
        f = d * a + b
        return np.where(f >= 0, np.exp(-f) / (1.0 + np.exp(-f)), 1.0 / (1.0 + np.exp(f)))

    @staticmethod
    def _couple(r: np.ndarray, max_iter: int = 100, eps: float = 0.005) -> np.ndarray:
        """Wu-Lin-Weng pairwise coupling (libsvm `multiclass_probability`)."""
        k = r.shape[0]
        p = np.full(k, 1.0 / k)
        Q = np.zeros((k, k))
        for t in range(k):
            Q[t, t] = sum(r[j, t] ** 2 for j in range(k) if j != t)
            for j in range(k):
                if j != t:
                    Q[t, j] = -r[j, t] * r[t, j]
        for _ in range(max_iter):
            Qp = Q @ p
            pQp = p @ Qp
            if np.max(np.abs(Qp - pQp)) < eps * abs(pQp) + 1e-12:
                break
            for t in range(k):
                diff = (-Qp[t] + pQp) / Q[t, t]
                p[t] += diff
                pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) ** 2
                Qp = (Qp + diff * Q[:, t]) / (1 + diff)
                p /= (1 + diff)
        return p

    def predict_proba(self, X) -> np.ndarray:
        dec = self._pairwise_decision(X)
        n_cls = len(self.classes_)
        out = np.zeros((dec.shape[0], n_cls))
        lo, hi = self._MIN_P, 1.0 - self._MIN_P
        for s in range(dec.shape[0]):
            r = np.zeros((n_cls, n_cls))
            for k, (i, j) in enumerate(self._pairs):
                pij = float(self._platt(dec[s, k], self.probA_[k], self.probB_[k]))
                pij = min(max(pij, lo), hi)
                r[i, j] = pij
                r[j, i] = 1.0 - pij
            out[s] = self._couple(r)
        return out


def load_models(path: Path | str | None = None) -> dict:
    """Load both classifiers. Returns the same {name: estimator} shape the app expects.

    Key order matches the original pickle: logistic regression first, SVM second.
    """
    p = Path(path) if path is not None else MODEL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Generate it with: python tools/export_model_params.py"
        )
    d = np.load(p, allow_pickle=False)
    return {
        "model1": LogisticRegressionOvR(d["lr_coef"], d["lr_intercept"], d["lr_classes"]),
        "model2": LinearSVCOvO(d["sv_coef"], d["sv_intercept"],
                               d["sv_probA"], d["sv_probB"], d["sv_classes"]),
    }
