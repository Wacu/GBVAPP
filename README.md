# GBVAPP
An App to allow users to get predictions of forms of GBV from text data

## Forms of GBV
1. Sexual Violence
2. Emotional Violence
3. Economic Violence
4. Harmful Traditional Practices
5. Physical Violence

## The App will give users the following interactions
1. Fetch data from twitter based on location
2. Download the csv file \ Store in SQLite DB
3. Clean the data
4. View sentiments
5. View main topics/themes from the data
6. Get the predicted forms of GBV
7. Visually see the distribution of the forms on a MAP

## Getting Started

### Clone the repository
```
git clone https://github.com/Wacu/GBVAPP.git
cd GBVAPP
```

### Install dependencies

Requires **Python 3.10+** (developed and verified on 3.12).
```
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` lists direct dependencies only. For a reproducible install
of the exact versions verified to work, use the lockfile instead:

```
pip install -r requirements-lock.txt
```


### Run the app
```
streamlit run Home.py
```



## Configuration

Twitter/X API credentials are read from Streamlit secrets, never committed:

```
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then fill in the values
```

On Streamlit Community Cloud, paste the same contents under **Settings -> Secrets**
rather than committing a file. `.streamlit/secrets.toml` is gitignored.

## Known issues

- **The trained models no longer depend on a pickle.** `models.joblib` was
  written by scikit-learn 0.23.1 (2020); loading it on a modern scikit-learn
  raised `InconsistentVersionWarning`, and the risk was real -- scikit-learn
  >= 1.7 removed `LogisticRegression(multi_class=...)`, so a one-vs-rest model
  had its `predict_proba` served as softmax. `probability=True` for SVC is also
  deprecated and disappears in 1.11, which would have removed
  `SVC.predict_proba` outright.

  Inference now reads `models.npz` (6 KB) via `apps/gbv_models.py`, which
  evaluates the learned weights directly in numpy. `predict` is byte-identical
  to the legacy pickle on 8,000 test vectors; the logistic regression's
  `predict_proba` is deliberately computed one-vs-rest, as trained. Regenerate
  with `python tools/export_model_params.py` and check with
  `python tools/verify_model_equivalence.py` (add `--real` to verify against
  tweets embedded from `gbv.db` rather than synthetic vectors; that pass needs
  the GloVe model, so it is opt-in).

  **This preserves the existing model faithfully; it does not make it correct.**
  The models are still 2020 artifacts of unknown accuracy, and the repo contains
  no labelled data to evaluate or retrain them (`gbv.db` holds 567,123
  unlabelled tweets and 500 cleaned ones, with no form label). Retraining on a
  labelled set remains the real fix. Note the models predict **4** classes,
  while this README lists **5** forms of GBV -- Harmful Traditional Practices is
  not among them.
- **The Twitter/X ingestion path is non-functional.** It targeted API v1.1
  `search_tweets`, which lost free access in 2023. The reference implementation
  is retained (commented out) in `apps/functions.py` and would need rewriting
  against API v2.
- **`gbv.db` (114 MB) is tracked via Git LFS.** A committed SQLite file cannot
  persist writes across restarts on a hosted platform; move to a managed
  database before deploying anything that writes.
- `apps/topics.py` is not imported anywhere and is effectively dead code.
