"""One-time conversion of models.joblib -> models.npz.

Why this exists
---------------
`models.joblib` was pickled with scikit-learn 0.23.1 (2020). Loading it on any
modern scikit-learn raises `InconsistentVersionWarning`, and the warning is not
cosmetic: scikit-learn 1.7+ removed `LogisticRegression(multi_class=...)`, so a
model trained one-vs-rest has its `predict_proba` computed as softmax instead.

Rather than keep a pickle whose behaviour depends on the installed library
version, this script extracts the learned parameters into a plain `.npz` of
numpy arrays. `apps/gbv_models.py` then evaluates them explicitly, so inference
is pinned to the maths the models were trained with and cannot drift again.

Both estimators are linear, so only the weight matrices are needed:
  * LogisticRegression  -> coef_, intercept_, classes_          (one-vs-rest)
  * SVC(kernel='linear') -> coef_, intercept_, probA_, probB_    (one-vs-one)

The SVC's 11,334 support vectors are NOT needed: for a linear kernel the
decision function collapses to `X @ coef_.T + intercept_`. Dropping them is
what takes the artifact from ~5 MB to a few KB.

Usage:  python tools/export_model_params.py
Run from the repository root. Requires the legacy pickle to still be loadable.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models.joblib"
DST = ROOT / "models.npz"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")

    # The version warning is the entire reason this script exists; silence it
    # here only so the output stays readable.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        models = joblib.load(SRC)

    keys = list(models.keys())
    if len(keys) != 2:
        raise SystemExit(f"expected 2 estimators, found {keys}")
    lr, sv = models[keys[0]], models[keys[1]]

    if sv.kernel != "linear":
        raise SystemExit(
            f"SVC kernel is {sv.kernel!r}, not 'linear'. This exporter drops "
            "support vectors, which is only valid for a linear kernel."
        )

    out = {
        # names are prefixed so a single .npz holds both estimators
        "lr_coef": np.asarray(lr.coef_, dtype=np.float64),
        "lr_intercept": np.asarray(lr.intercept_, dtype=np.float64),
        "lr_classes": np.asarray(lr.classes_),
        "sv_coef": np.asarray(sv.coef_, dtype=np.float64),
        "sv_intercept": np.asarray(sv.intercept_, dtype=np.float64),
        "sv_probA": np.asarray(sv.probA_, dtype=np.float64),
        "sv_probB": np.asarray(sv.probB_, dtype=np.float64),
        "sv_classes": np.asarray(sv.classes_),
        # provenance, so the artifact is self-describing
        "meta_source_sklearn": np.asarray("0.23.1"),
        "meta_lr_multi_class": np.asarray("ovr"),
        "meta_sv_decision": np.asarray("ovo"),
        "meta_n_features": np.asarray(int(lr.n_features_in_)),
    }

    np.savez_compressed(DST, **out)

    print(f"wrote {DST}  ({DST.stat().st_size / 1024:.1f} KB)")
    print(f"  source: {SRC.name} ({SRC.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  dropped {sv.support_vectors_.shape[0]} support vectors "
          "(unused for a linear kernel)")
    print(f"  classes: {list(lr.classes_)}   features: {lr.n_features_in_}")


if __name__ == "__main__":
    main()
