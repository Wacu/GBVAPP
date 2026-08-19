"""Check models.npz reproduces models.joblib, and quantify where it deliberately differs.

    python tools/verify_model_equivalence.py                # synthetic vectors only (fast)
    python tools/verify_model_equivalence.py --real         # also real embedded tweets
    python tools/verify_model_equivalence.py --real --no-synthetic
    python tools/verify_model_equivalence.py --real --rows 100

Why two datasets
----------------
The synthetic pass sweeps `normal(0, s)` at several scales, which covers the
input space cheaply and needs neither the GloVe download nor the database. It
is not, however, drawn from the distribution the app actually sees.

The `--real` pass embeds tweets from `gbv.db` through the app's own
`FunctionText2Vec`, so the models are exercised on genuine feature vectors.
It needs the 66 MB `glove-wiki-gigaword-50` model (cached after first use) and
is therefore opt-in.

What must hold, on every dataset
--------------------------------
* `predict` must match the legacy pickle EXACTLY for both estimators. This is
  the app's actual output (the predicted GBV form), so any difference is a
  regression.
* SVM `predict_proba` must match within libsvm's own convergence tolerance.
  The coupling algorithm stops at eps=0.005 relative, so bit-equality is not
  meaningful; an argmax flip is tolerated only where the top two classes are
  tied to within that same tolerance.
* Logistic-regression `predict_proba` is EXPECTED to differ from the pickle as
  served by modern scikit-learn: the pickle records multi_class='ovr' but
  scikit-learn >=1.7 scores it as softmax. This script reports the size of that
  gap rather than asserting equality.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from apps.gbv_models import load_models  # noqa: E402

N_PER_SCALE = 2000
SCALES = (0.5, 2.0, 8.0, 20.0)
TOL = 0.005          # libsvm's own relative stopping tolerance
ROWSUM_TOL = 1e-9


def synthetic_matrix() -> np.ndarray:
    rng = np.random.default_rng(20260819)
    return np.vstack([rng.normal(0, s, size=(N_PER_SCALE, 50)) for s in SCALES])


def real_matrix(limit: int | None = None) -> np.ndarray:
    """Embed real cleaned tweets through the app's own feature pipeline.

    Imported lazily: this pulls in streamlit and gensim and may download GloVe,
    none of which the synthetic pass needs.
    """
    import sqlite3

    import pandas as pd

    db = ROOT / "gbv.db"
    if not db.exists():
        raise FileNotFoundError(f"{db} not found; cannot build the real-data matrix")

    from apps.models import FunctionText2Vec

    # apps.models decorates its loader with st.cache_resource, which logs a
    # bare-mode warning outside a Streamlit run. Streamlit configures its own
    # loggers when imported, so this has to run after the import above to stick.
    for name in list(logging.root.manager.loggerDict):
        if name == "streamlit" or name.startswith("streamlit."):
            logging.getLogger(name).setLevel(logging.ERROR)

    con = sqlite3.connect(db)
    try:
        sql = "SELECT lemma_nostops FROM cleaned"
        if limit:
            sql += f" LIMIT {int(limit)}"
        text = pd.read_sql(sql, con)["lemma_nostops"].fillna("")
    finally:
        con.close()

    if text.empty:
        raise ValueError("the 'cleaned' table returned no rows")

    print(f"   embedding {len(text)} real tweets via FunctionText2Vec "
          "(first run downloads GloVe, ~66 MB) ...")
    return FunctionText2Vec(text).to_numpy(dtype=np.float64)


def run_checks(legacy: dict, new: dict, X: np.ndarray, dataset: str) -> list[str]:
    """Run every equivalence check over one feature matrix. Returns failures."""
    failures: list[str] = []
    lkeys = list(legacy.keys())
    print(f"\n===== {dataset}: {X.shape[0]} vectors x {X.shape[1]} features")
    print(f"      feature range [{X.min():.2f}, {X.max():.2f}]")

    def fail(msg: str) -> None:
        failures.append(f"[{dataset}] {msg}")

    # ---- predict: must be exact for both estimators ----
    for label, lk, nk in (("LogisticRegression", lkeys[0], "model1"),
                          ("LinearSVC", lkeys[1], "model2")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = legacy[lk].predict(X)
        got = new[nk].predict(X)
        n_diff = int((ref != got).sum())
        print(f"[{'PASS' if n_diff == 0 else 'FAIL'}] {label}.predict  exact match  "
              f"({X.shape[0] - n_diff}/{X.shape[0]})")
        if n_diff:
            fail(f"{label}.predict differs on {n_diff} rows")

    # ---- SVM predict_proba: within libsvm tolerance, flips only on ties ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_p = legacy[lkeys[1]].predict_proba(X)
    got_p = new["model2"].predict_proba(X)
    dev = float(np.abs(ref_p - got_p).max())
    disagree = np.where(ref_p.argmax(1) != got_p.argmax(1))[0]
    if len(disagree):
        srt = np.sort(ref_p[disagree], axis=1)[:, ::-1]
        gaps = srt[:, 0] - srt[:, 1]
    else:
        gaps = np.array([])
    decisive = disagree[gaps >= TOL] if len(disagree) else np.array([], dtype=int)
    ok = dev < TOL and len(decisive) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] LinearSVC.predict_proba  max|dev|={dev:.2e} "
          f"(tol {TOL:g}); argmax flips={len(disagree)}, "
          f"all within tie tolerance={len(decisive) == 0}")
    if len(disagree):
        print(f"         flipped rows are ties; largest top-2 gap {gaps.max():.2e} "
              f"vs tolerance {TOL:g}")
    if not ok:
        fail(f"SVM predict_proba: deviation {dev:.2e}, "
             f"{len(decisive)} flips outside tie tolerance")

    rowsum = float(np.abs(got_p.sum(1) - 1.0).max())
    print(f"[{'PASS' if rowsum < ROWSUM_TOL else 'FAIL'}] LinearSVC.predict_proba  "
          f"rows sum to 1 (max err {rowsum:.1e})")
    if rowsum >= ROWSUM_TOL:
        fail(f"SVM predict_proba rows do not sum to 1 (err {rowsum:.1e})")

    # ---- LogReg predict_proba: intentional divergence, quantified ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        served = legacy[lkeys[0]].predict_proba(X)   # softmax, as modern sklearn serves it
        D = legacy[lkeys[0]].decision_function(X)
    ours = new["model1"].predict_proba(X)            # one-vs-rest, as trained

    sm = np.exp(D - D.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    print("       LogisticRegression.predict_proba -- intentional scheme change:")
    print(f"         legacy as served by sklearn == softmax : "
          f"{np.abs(served - sm).max():.2e} deviation")
    print(f"         our one-vs-rest vs that softmax        : "
          f"{np.abs(ours - served).max():.3f} max deviation")
    print(f"         rows where the schemes' argmax differs : "
          f"{100 * (ours.argmax(1) != served.argmax(1)).mean():.2f}%")
    print("         (predict is unaffected -- argmax of the decision function)")

    rowsum_lr = float(np.abs(ours.sum(1) - 1.0).max())
    print(f"[{'PASS' if rowsum_lr < ROWSUM_TOL else 'FAIL'}] "
          f"LogisticRegression.predict_proba  rows sum to 1 (max err {rowsum_lr:.1e})")
    if rowsum_lr >= ROWSUM_TOL:
        fail(f"LogReg predict_proba rows do not sum to 1 (err {rowsum_lr:.1e})")

    # Class coverage is not a pass/fail condition, but a matrix that never
    # reaches a class proves nothing about that class's decision region.
    u, c = np.unique(new["model1"].predict(X), return_counts=True)
    print(f"       classes exercised (LogReg): {dict(zip(u.tolist(), c.tolist()))}")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true",
                    help="also verify against real tweets embedded from gbv.db")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="skip the synthetic pass (only useful with --real)")
    ap.add_argument("--rows", type=int, default=None,
                    help="limit the number of real tweets embedded (default: all)")
    args = ap.parse_args()

    if args.no_synthetic and not args.real:
        ap.error("--no-synthetic leaves nothing to run; add --real")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import joblib
        legacy = joblib.load(ROOT / "models.joblib")
    new = load_models(ROOT / "models.npz")

    failures: list[str] = []
    ran: list[str] = []

    if not args.no_synthetic:
        failures += run_checks(legacy, new, synthetic_matrix(), "SYNTHETIC")
        ran.append("synthetic")

    if args.real:
        try:
            X = real_matrix(args.rows)
        except Exception as e:
            print(f"\n===== REAL DATA: could not build matrix -- "
                  f"{type(e).__name__}: {e}")
            failures.append(f"[REAL] could not build matrix: {e}")
        else:
            failures += run_checks(legacy, new, X, "REAL DATA")
            ran.append("real")

    print()
    if failures:
        print(f"RESULT: FAILED ({', '.join(ran) or 'nothing'} checked)")
        for f in failures:
            print("  -", f)
        return 1
    print(f"RESULT: models.npz reproduces models.joblib on {' and '.join(ran)} data.")
    if not args.real:
        print("        (synthetic only -- rerun with --real to check actual tweets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
