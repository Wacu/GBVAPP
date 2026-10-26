"""Tests for the batch GBV-form prediction path.

The cleaned-data branch of `pages/03.GBV form detect.py` never worked: it read
the table into a variable it discarded, predicted on an empty list, and
displayed row 0 of the result as though it answered for every row. These tests
pin the behaviour the fixed code must have.

The model is faked throughout -- the real estimators need the 66 MB GloVe
vectors, and none of the logic under test depends on them.
"""

import numpy as np
import pandas as pd
import pytest

from apps.predict import FORM_LABELS, confidence_for, label_for, labels_for, predict_batch


class FakeModel:
    """Minimal stand-in exposing the scikit-learn surface the app uses.

    `classes_` is deliberately not 0..n-1 in some tests: the confidence lookup
    must find the predicted label's column rather than assume label == index.
    """

    def __init__(self, predictions, probabilities, classes=(0, 1, 2, 3)):
        self._predictions = np.asarray(predictions)
        self._probabilities = np.asarray(probabilities, dtype=float)
        self.classes_ = np.asarray(classes)

    def predict(self, X):
        return self._predictions

    def predict_proba(self, X):
        return self._probabilities


def fake_vectorise(texts):
    """Stand in for FunctionText2Vec: one 50-dim row per input text."""
    return pd.DataFrame(np.zeros((len(texts), 50)))


# --- label mapping ---------------------------------------------------------

def test_label_for_maps_each_trained_class():
    assert label_for(0) == "Economic Violence"
    assert label_for(1) == "Emotional Violence"
    assert label_for(2) == "Physical Violence"
    assert label_for(3) == "Sexual Violence"


def test_label_for_unknown_class_is_not_silently_a_form():
    # The models emit only 0-3. Anything else must not be dressed up as a form.
    assert label_for(9) == "No violence detected"


def test_labels_for_maps_every_row_not_just_the_first():
    out = labels_for(np.array([3, 0, 2]))
    assert out == ["Sexual Violence", "Economic Violence", "Physical Violence"]


def test_labels_for_handles_an_empty_batch():
    assert labels_for(np.array([])) == []


# --- confidence ------------------------------------------------------------

def test_confidence_is_for_the_predicted_label_not_the_max():
    # SVM predict() (voting) and predict_proba() (Platt) can disagree; the
    # number shown must describe the label actually displayed.
    model = FakeModel(predictions=[0], probabilities=[[0.3, 0.7, 0.0, 0.0]])
    assert confidence_for(model, model.predict(None), model.predict_proba(None)) == [30.0]


def test_confidence_looks_up_by_class_value_not_column_index():
    # classes_ are 1..4, so class 3 lives in column 2, not column 3.
    model = FakeModel(
        predictions=[3],
        probabilities=[[0.1, 0.2, 0.6, 0.1]],
        classes=(1, 2, 3, 4),
    )
    assert confidence_for(model, model.predict(None), model.predict_proba(None)) == [60.0]


def test_confidence_returns_one_value_per_row():
    model = FakeModel(
        predictions=[0, 1, 2],
        probabilities=[[0.9, 0.1, 0.0, 0.0],
                       [0.2, 0.8, 0.0, 0.0],
                       [0.0, 0.0, 0.5, 0.5]],
    )
    assert confidence_for(model, model.predict(None), model.predict_proba(None)) == [90.0, 80.0, 50.0]


# --- batch prediction ------------------------------------------------------

def test_predict_batch_returns_one_row_per_input():
    texts = ["he beat her", "she was fired for refusing", "constant insults"]
    model = FakeModel(
        predictions=[2, 0, 1],
        probabilities=[[0.0, 0.0, 0.8, 0.2],
                       [0.7, 0.1, 0.1, 0.1],
                       [0.1, 0.6, 0.2, 0.1]],
    )
    out = predict_batch(texts, model, vectorise=fake_vectorise)

    assert len(out) == len(texts)
    assert list(out["text"]) == texts
    assert list(out["form"]) == ["Physical Violence", "Economic Violence", "Emotional Violence"]
    assert list(out["confidence"]) == [80.0, 70.0, 60.0]


def test_predict_batch_rejects_an_empty_batch_with_a_clear_error():
    # The old code passed [] straight to CountVectorizer, which raised an
    # opaque "empty vocabulary" ValueError on submit.
    with pytest.raises(ValueError, match="no texts"):
        predict_batch([], FakeModel([], np.empty((0, 4))), vectorise=fake_vectorise)


def test_predict_batch_drops_blank_texts_rather_than_crashing():
    # Rows whose cleaning removed every token reach us as '' or NaN. They carry
    # no signal, so they must not be fed to the vectoriser.
    texts = ["he beat her", "", None, "   ", np.nan, "she was fired"]
    model = FakeModel(
        predictions=[2, 0],
        probabilities=[[0.0, 0.0, 0.8, 0.2], [0.7, 0.1, 0.1, 0.1]],
    )
    out = predict_batch(texts, model, vectorise=fake_vectorise)

    assert list(out["text"]) == ["he beat her", "she was fired"]


def test_predict_batch_errors_when_every_text_is_blank():
    with pytest.raises(ValueError, match="no texts"):
        predict_batch(["", "  ", None], FakeModel([], np.empty((0, 4))), vectorise=fake_vectorise)


def test_form_labels_covers_only_the_four_trained_classes():
    # A guard on the README's 4-vs-5 discrepancy: Harmful Traditional Practices
    # is documented but not trainable here, and must not appear.
    assert set(FORM_LABELS) == {0, 1, 2, 3}
    assert "Harmful Traditional Practices" not in FORM_LABELS.values()
