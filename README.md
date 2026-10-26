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

On first run the app downloads the data it needs and caches it for later runs:

- NLTK corpora `punkt_tab`, `stopwords` and `wordnet` (handled by
  `apps/nltk_setup.py`; a few MB)
- the `glove-wiki-gigaword-50` word vectors, on first use of the GBV-form
  page (~66 MB)

Note nltk 3.9 renamed the tokenizer data from `punkt` to `punkt_tab`, so a
machine that has run older versions of this app may hold `punkt` and still need
the new resource.



## Deploying

The build needs **Python 3.12 or 3.13**, pinned by the `.python-version` file in
the repo root. This is not cosmetic: `gensim` publishes wheels only up to
cp313, so on a newer interpreter pip falls back to compiling it from source and
the build fails against changed CPython internals
(`'PyLongObject' has no member named 'ob_digit'`). Every other compiled
dependency here already has cp314 wheels -- gensim is the only one holding the
version back.

On Render, set the start command so Streamlit binds the injected port:

```
streamlit run Home.py --server.port $PORT --server.address 0.0.0.0
```

Two constraints worth knowing before choosing a plan:

- **Memory.** The GloVe vectors alone occupy ~76 MB resident, on top of numpy,
  scipy, pandas, scikit-learn, matplotlib and Streamlit. A 512 MB instance is
  tight.
- **Ephemeral disk.** The NLTK corpora and the 66 MB GloVe model are downloaded
  at runtime, so any platform that resets its filesystem re-downloads them on
  every cold start.

## Configuration

Twitter/X API credentials are read from Streamlit secrets, never committed:

```
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then fill in the values
```

On Streamlit Community Cloud, paste the same contents under **Settings -> Secrets**
rather than committing a file. `.streamlit/secrets.toml` is gitignored.

## Security

Dependencies are pinned in `requirements-lock.txt` and audited with
[`pip-audit`](https://pypi.org/project/pip-audit/):

```
pip install pip-audit
pip-audit -r requirements-lock.txt
```

Re-run this after any dependency change, and record the result in the
lockfile header. GitHub's Dependabot alerts track the **default branch**
(`master`), so a fix only clears the alert once it lands there.

### Audit of 2026-10-26

46 advisories were open against the previous pinned set. 45 are cleared by
these bumps:

| Package | Was | Now | Advisories |
| --- | --- | --- | --- |
| pillow | 12.3.0 | 13.0.0 | 38, across the image decoders |
| urllib3 | 2.7.0 | 2.8.0 | 3 |
| certifi | 2026.7.22 | 2026.10.23 | 1 |

`Pillow>=13` requires `streamlit>=1.63`; streamlit 1.62 and earlier pin
`pillow<13`. The floor therefore moved to the verified `streamlit>=1.66.1`,
which also moved `pyarrow` and `starlette` in the lockfile.

Note that Dependabot's own grouped PR (`dependabot/pip/all-minor-patch-*`)
raises streamlit, pandas, numpy, matplotlib, scikit-learn and joblib --
**none of which are the vulnerable packages**. Merging it alone does not
clear these alerts.

### Accepted risk: nltk PYSEC-2026-3740

One advisory is knowingly unfixed, because no patched release exists. It is
a file-sandbox bypass in nltk's model-artifact APIs: `TransitionParser.train`
and `.parse`, `AveragedPerceptron.save` and `.load`,
`PerceptronTagger.save_to_json`, and `save_maxent_params` treat
caller-controlled model paths as ordinary filenames.

This app calls none of them. Its nltk surface is `word_tokenize`, `ngrams`,
`FreqDist`, `stopwords`, `WordNetLemmatizer`, `PorterStemmer` and the VADER
analyser, and it never accepts a model path from a user. Re-evaluate if that
changes, or when nltk ships a fix.

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
