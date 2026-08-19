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

- **`models.joblib` was pickled with scikit-learn 0.23.1** and now loads on a
  modern scikit-learn, which raises `InconsistentVersionWarning`. It loads and
  predicts, but scikit-learn does not guarantee correctness across that gap.
  The models should be **retrained and re-serialised** before their output is
  relied on for anything.
- **The Twitter/X ingestion path is non-functional.** It targeted API v1.1
  `search_tweets`, which lost free access in 2023. The reference implementation
  is retained (commented out) in `apps/functions.py` and would need rewriting
  against API v2.
- **`gbv.db` (114 MB) is tracked via Git LFS.** A committed SQLite file cannot
  persist writes across restarts on a hosted platform; move to a managed
  database before deploying anything that writes.
- `apps/topics.py` is not imported anywhere and is effectively dead code.
