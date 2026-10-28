"""Term counting for the word cloud and frequency charts.

Kept out of `apps/eda.py` so it can be tested without importing Streamlit or
rendering anything.

The cloud was previously built from the `lemma_nostops` column. That column is
the wrong input twice over: cleaning has already stripped the `@` and `#`
markers, so usernames are indistinguishable from words by the time it is
written, and `lem()` lemmatises every token as a verb, which leaves plural
nouns alone -- `woman` and `women` were counted as two different terms. This
reads the raw tweet instead, while those markers still exist.
"""

import re
from collections import Counter

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer

from apps.nltk_setup import ensure_nltk_data

ensure_nltk_data()

# The hashtags and keywords the corpus was collected by. 9.5% of all tokens in
# the `cleaned` table are one of these, and 457 of 499 rows contain
# `domesticviolence` -- so they describe the search, not the findings, and as
# the largest words in the cloud they crowd out everything informative.
CORPUS_TERMS = {
    'domesticviolence', 'domestic', 'violence', 'violent',
    'violenceagainstwomen', 'gbv', 'genderbasedviolence',
    'endgbv', 'saynotoviolence', 'abuse',
}

_URL = re.compile(r'https?://\S+|www\.\S+')
_HANDLE = re.compile(r'@\w+')
_HASHMARK = re.compile(r'#(\w+)')
# Letters only. Contractions split into fragments ("i'm" -> "i", "m") that the
# minimum-length rule then drops, which is what we want in a cloud.
_WORD = re.compile(r'[a-z]+')

_LEMMATIZER = WordNetLemmatizer()


def _base_stopwords():
    words = set(nltk_stopwords.words('english'))
    # Retweet and reply markers, plus filler that survives the stopword list
    # and says nothing about the subject.
    words.update({'rt', 'amp', 'via', 'u', 'ur', 'im', 'dont', 'doesnt',
                  'cant', 'wont', 'get', 'got', 'go', 'going', 'one', 'also',
                  'like', 'would', 'could', 'may', 'many', 'much', 'us'})
    return words


def singular(term):
    """Reduce a term to its singular noun form.

    `victims` -> `victim`, `children` -> `child`, `women` -> `woman`. This is
    the step `lem()` misses: it lemmatises as a verb, which leaves nouns
    untouched.
    """
    return _LEMMATIZER.lemmatize(term, pos='n')


def wordcloud_frequencies(texts, extra_stopwords=(), drop_corpus_terms=True):
    """Count display terms across raw tweets.

    Returns {term: count}, suitable for `WordCloud.generate_from_frequencies`.
    Counting here rather than letting WordCloud tokenise means the exact same
    numbers can drive the frequency bar charts, and can be tested.
    """
    stops = _base_stopwords()
    stops.update(singular(str(w).lower()) for w in extra_stopwords)
    if drop_corpus_terms:
        stops.update(CORPUS_TERMS)

    counts = Counter()
    for text in texts:
        if not text:
            continue
        text = str(text).lower()
        text = _URL.sub(' ', text)
        # Handles go before the '@' is lost. A username reads as an ordinary
        # word once cleaning has stripped the marker.
        text = _HANDLE.sub(' ', text)
        # Keep the hashtag's text -- '#domesticviolence' is a real term, it is
        # just one that happens to be the query.
        text = _HASHMARK.sub(r'\1', text)

        for match in _WORD.findall(text):
            # Stopwords are checked before lemmatising as well as after.
            # Lemmatising first turns "does" into "doe" -- the deer -- which
            # no longer matches the stopword list and reached the cloud.
            if len(match) < 3 or match in stops:
                continue
            term = singular(match)
            if term in stops or len(term) < 3:
                continue
            counts[term] += 1

    return dict(counts)


def top_ngrams(texts, n=1, top=10, extra_stopwords=(), drop_corpus_terms=True):
    """Most frequent n-grams, built from the same filtered token stream.

    The existing charts tokenise `lemma_nostops` independently, so they and the
    cloud can disagree about what is frequent. This shares the filtering.
    """
    stops = _base_stopwords()
    stops.update(singular(str(w).lower()) for w in extra_stopwords)
    if drop_corpus_terms:
        stops.update(CORPUS_TERMS)

    counts = Counter()
    for text in texts:
        if not text:
            continue
        text = str(text).lower()
        text = _URL.sub(' ', text)
        text = _HANDLE.sub(' ', text)
        text = _HASHMARK.sub(r'\1', text)

        tokens = [singular(m) for m in _WORD.findall(text)
                  if len(m) >= 3 and m not in stops]
        tokens = [t for t in tokens if t not in stops and len(t) >= 3]
        counts.update(' '.join(gram) for gram in nltk.ngrams(tokens, n))

    return counts.most_common(top)
