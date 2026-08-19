"""Check models.npz reproduces models.joblib, and quantify where it deliberately differs.

Run after regenerating models.npz:  python tools/verify_model_equivalence.py

What must hold
--------------
* `predict` must match the legacy pickle EXACTLY for both estimators. This is
  the app's actual output (the predicted GBV form), so any difference is a
  regression.
* SVM `predict_proba` must match within libsvm's own convergence tolerance
  (the coupling algorithm stops at eps=0.005 relative, so bit-equality is not
  meaningful).
* Logistic-regression `predict_proba` is EXPECTED to differ from the pickle as
  served by modern scikit-learn: the pickle records multi_class='ovr' but
  scikit-learn >=1.7 scores it as softmax. This script reports the size of that
  gap rather than asserting equality.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apps.gbv_models import load_models  # noqa: E402

N_PER_SCALE = 2000
SCALES = (0.5, 2.0, 8.0, 20.0)   # GloVe-50 sums span a wide range; cover it


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import joblib
        legacy = joblib.load(root / "models.joblib")
    new = load_models(root / "models.npz")

    rng = np.random.default_rng(20260819)
    X = np.vstack([rng.normal(0, s, size=(N_PER_SCALE, 50)) for s in SCALES])
    print(f"test matrix: {X.shape[0]} vectors x {X.shape[1]} features\n")

    failures = []
    lkeys = list(legacy.keys())

    # ---- predict: must be exact ----
    for label, lk, nk in (("LogisticRegression", lkeys[0], "model1"),
                          ("LinearSVC", lkeys[1], "model2")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = legacy[lk].predict(X)
        got = new[nk].predict(X)
        n_diff = int((ref != got).sum())
        status = "PASS" if n_diff == 0 else "FAIL"
        print(f"[{status}] {label}.predict  exact match  "
              f"({X.shape[0] - n_diff}/{X.shape[0]})")
        if n_diff:
            failures.append(f"{label}.predict differs on {n_diff} rows")

    # ---- SVM predict_proba: within libsvm tolerance ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref_p = legacy[lkeys[1]].predict_proba(X)
    got_p = new["model2"].predict_proba(X)
    dev = float(np.abs(ref_p - got_p).max())
    disagree = np.where(ref_p.argmax(1) != got_p.argmax(1))[0]
    # An argmax flip is only acceptable where the classes are tied to within the
    # coupling algorithm's own stopping tolerance -- there the ordering is not
    # meaningful in either implementation. A flip on a clear winner is a defect.
    TOL = 0.005
    srt = np.sort(ref_p[disagree], axis=1)[:, ::-1] if len(disagree) else np.empty((0, 2))
    gaps = (srt[:, 0] - srt[:, 1]) if len(disagree) else np.array([])
    decisive = disagree[gaps >= TOL] if len(disagree) else np.array([], dtype=int)
    ok = dev < TOL and len(decisive) == 0
    print(f"[{'PASS' if ok else 'FAIL'}] LinearSVC.predict_proba  "
          f"max|dev|={dev:.2e} (tol {TOL:g}); argmax flips={len(disagree)}, "
          f"all within tie tolerance={len(decisive) == 0}")
    if len(disagree):
        print(f"         (flipped rows are ties; largest top-2 gap "
              f"{gaps.max():.2e} vs tolerance {TOL:g})")
    if not ok:
        failures.append(
            f"SVM predict_proba: deviation {dev:.2e}, "
            f"{len(decisive)} flips outside tie tolerance")

    rowsum = float(np.abs(got_p.sum(1) - 1.0).max())
    print(f"[{'PASS' if rowsum < 1e-9 else 'FAIL'}] LinearSVC.predict_proba  "
          f"rows sum to 1 (max err {rowsum:.1e})")

    # ---- LogReg predict_proba: intentional divergence, quantified ----
    print()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        served = legacy[lkeys[0]].predict_proba(X)     # softmax, as modern sklearn serves it
        D = legacy[lkeys[0]].decision_function(X)
    ours = new["model1"].predict_proba(X)              # one-vs-rest, as trained

    sm = np.exp(D - D.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    print("LogisticRegression.predict_proba -- intentional scheme change:")
    print(f"   legacy pickle as served by sklearn == softmax : "
          f"{np.abs(served - sm).max():.2e} deviation")
    print(f"   our one-vs-rest vs that softmax               : "
          f"{np.abs(ours - served).max():.3f} max deviation")
    print(f"   rows where the two schemes' argmax disagrees  : "
          f"{100 * (ours.argmax(1) != served.argmax(1)).mean():.2f}%")
    print(f"   (predict is unaffected -- it is argmax of the decision function)")
    rowsum_lr = float(np.abs(ours.sum(1) - 1.0).max())
    print(f"[{'PASS' if rowsum_lr < 1e-9 else 'FAIL'}] our rows sum to 1 "
          f"(max err {rowsum_lr:.1e})")

    print()
    if failures:
        print("RESULT: FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("RESULT: models.npz reproduces models.joblib for all predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
