"""Ensure the NLTK data files this app needs are present.

NLTK ships code but no corpora -- they are downloaded at runtime into a
user-level directory. This app previously relied on the developer's machine
already having them, so a fresh checkout (or any hosted deployment, which
starts with an empty home directory) failed at the first tokenisation with
`LookupError: Resource 'punkt_tab' not found`.

Note that nltk 3.9 renamed the punkt tokenizer data from `punkt` to
`punkt_tab`. A machine that has been running the app for years can therefore
hold `punkt` and still fail on a modern nltk, which is exactly what happened
here. Checking by resource path rather than by "have I ever downloaded punkt"
is what makes this robust across that rename.
"""

from __future__ import annotations

import nltk

# (resource path passed to nltk.data.find, download package name)
REQUIRED: tuple[tuple[str, str], ...] = (
    ("tokenizers/punkt_tab", "punkt_tab"),  # nltk.word_tokenize on nltk >= 3.9
    ("corpora/stopwords", "stopwords"),     # stopwords.words("english")
    ("corpora/wordnet", "wordnet"),         # WordNetLemmatizer
)

_checked = False


def ensure_nltk_data(quiet: bool = True) -> list[str]:
    """Download any missing NLTK resource. Returns the packages fetched.

    Cheap to call repeatedly: nltk.data.find is a filesystem lookup, and after
    the first successful pass this short-circuits for the life of the process.
    """
    global _checked
    if _checked:
        return []

    fetched = []
    for path, package in REQUIRED:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=quiet)
            fetched.append(package)

    _checked = True
    return fetched


if __name__ == "__main__":
    got = ensure_nltk_data(quiet=False)
    print(f"downloaded: {got}" if got else "all NLTK resources already present")
