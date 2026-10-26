"""Batch prediction of GBV forms.

Extracted from `pages/03.GBV form detect.py`, where the logic was defined twice
and only ever worked for a single text. The cleaned-data branch read its table
into a discarded variable, predicted on an empty list, and displayed row 0 as
though it answered for the whole batch.

Keeping this in `apps/` makes it importable and testable; the Streamlit page
calls `st.set_page_config` at import time, so nothing in it can be exercised
from a test.
"""

import numpy as np
import pandas as pd

from apps.models import FunctionText2Vec

# The trained estimators emit these four classes only. The README documents a
# fifth form, Harmful Traditional Practices, which these models were never
# trained to predict -- it is deliberately absent rather than mapped onto a
# neighbouring class.
FORM_LABELS = {
    0: 'Economic Violence',
    1: 'Emotional Violence',
    2: 'Physical Violence',
    3: 'Sexual Violence',
}

NO_VIOLENCE = 'No violence detected'


def label_for(predicted):
    """Name the form for one predicted class value.

    Anything outside the trained classes falls through to NO_VIOLENCE rather
    than being presented as one of the four forms.
    """
    return FORM_LABELS.get(int(predicted), NO_VIOLENCE)


def labels_for(predicted):
    """Name the form for every row in a prediction array."""
    return [label_for(p) for p in np.asarray(predicted).ravel()]


def confidence_for(model, predicted, proba):
    """Probability the model assigned to the class it actually predicted, per row.

    Looks the predicted label up in `model.classes_` rather than assuming the
    class value equals its column index. For the SVM, predict() (one-vs-one
    voting) and predict_proba() (Platt coupling) can disagree, so this is not
    always the row maximum -- it is the confidence in the label being shown,
    which is what the caller wants.
    """
    classes = np.asarray(model.classes_)
    proba = np.asarray(proba, dtype=float)

    confidences = []
    for row, cls in enumerate(np.asarray(predicted).ravel()):
        matches = np.where(classes == cls)[0]
        if matches.size == 0:
            # Predicted a class the probability matrix has no column for; no
            # honest confidence exists for it.
            confidences.append(float('nan'))
            continue
        confidences.append(float(np.round(proba[row][int(matches[0])] * 100, 0)))
    return confidences


def _usable(texts):
    """Drop rows that carry no tokens.

    Cleaning strips every non-letter, so a tweet of only punctuation, a URL or
    emoji arrives here as '' or NaN. CountVectorizer raises an opaque "empty
    vocabulary" error on those, so they are removed before vectorising.
    """
    kept = []
    for text in texts:
        if text is None:
            continue
        if isinstance(text, float) and np.isnan(text):
            continue
        text = str(text).strip()
        if text:
            kept.append(text)
    return kept


def predict_batch(texts, model, vectorise=FunctionText2Vec):
    """Predict the GBV form for every text, returning one row per input.

    `vectorise` is injectable so tests need not download the 66 MB GloVe
    vectors; the default is the embedding the trained models expect.

    Returns a DataFrame with columns: text, form, confidence.
    """
    usable = _usable(texts)
    if not usable:
        raise ValueError(
            "no texts to predict on -- every row was empty after cleaning"
        )

    X = vectorise(usable)
    predicted = model.predict(X)
    proba = model.predict_proba(X)

    return pd.DataFrame({
        'text': usable,
        'form': labels_for(predicted),
        'confidence': confidence_for(model, predicted, proba),
    })
