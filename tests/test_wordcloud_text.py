"""Tests for the word-cloud term counting.

The cloud used to be built from `lemma_nostops` joined into one string and
handed straight to WordCloud. Three things made it unreadable, and all three
are properties of the *text*, not the styling:

  - the terms the corpus was collected by dominated it (9.5% of all tokens),
  - singulars and plurals were counted as different words, so `woman` (17) and
    `women` (87) appeared as two mid-sized words instead of one large one,
  - usernames survived cleaning and read as ordinary words.

These tests pin the counting. Rendering is left to `apps/eda.cloud`.
"""

import pytest

from apps.wordcloud_text import CORPUS_TERMS, wordcloud_frequencies


def test_drops_the_terms_the_corpus_was_collected_by():
    # Every row was collected by this hashtag, so it describes the query and
    # not the data. It must not be the biggest word in the picture.
    freqs = wordcloud_frequencies(['#domesticviolence she needed help',
                                   '#domesticviolence he was arrested'])
    assert 'domesticviolence' not in freqs
    assert freqs['help'] == 1


def test_keeps_corpus_terms_when_asked():
    freqs = wordcloud_frequencies(['#domesticviolence help'],
                                  drop_corpus_terms=False)
    assert freqs['domesticviolence'] == 1


def test_counts_singular_and_plural_as_one_term():
    freqs = wordcloud_frequencies(['the victim spoke', 'two victims spoke',
                                   'more victims spoke'])
    assert freqs['victim'] == 3
    assert 'victims' not in freqs


def test_collapses_irregular_plurals():
    freqs = wordcloud_frequencies(['one woman', 'many women', 'the children',
                                   'a child'])
    assert freqs['woman'] == 2
    assert freqs['child'] == 2


def test_drops_usernames():
    # Cleaning strips the @, after which a handle is indistinguishable from a
    # word -- so handles have to be removed while the marker is still there.
    freqs = wordcloud_frequencies(['thank you @fenwickwest for the support'])
    assert 'fenwickwest' not in freqs
    assert freqs['support'] == 1


def test_drops_urls():
    freqs = wordcloud_frequencies(['report it here https://example.org/help'])
    assert freqs['report'] == 1
    assert not any('http' in term or 'example' in term for term in freqs)


def test_drops_stopwords_and_punctuation():
    freqs = wordcloud_frequencies(['she was in the house, and it was dark!'])
    for stopword in ('she', 'was', 'in', 'the', 'and', 'it'):
        assert stopword not in freqs
    assert freqs['house'] == 1
    assert freqs['dark'] == 1


def test_extra_stopwords_are_honoured():
    # The page offers these so an analyst can drop a term that is swamping a
    # particular view without editing the code.
    freqs = wordcloud_frequencies(['help and support', 'help again'],
                                  extra_stopwords=['help'])
    assert 'help' not in freqs
    assert freqs['support'] == 1


def test_extra_stopwords_match_regardless_of_case_or_plurality():
    freqs = wordcloud_frequencies(['Victims spoke out'],
                                  extra_stopwords=['VICTIM'])
    assert 'victim' not in freqs


def test_empty_input_returns_no_terms_rather_than_raising():
    assert wordcloud_frequencies([]) == {}
    assert wordcloud_frequencies(['', '   ', None]) == {}


def test_stopwords_are_caught_before_lemmatising():
    # "does" lemmatises to "doe" -- a deer -- which no longer matches the
    # stopword list. Filtering only after lemmatising put it in the cloud.
    freqs = wordcloud_frequencies(['she does what she needs'])
    assert 'doe' not in freqs
    assert 'does' not in freqs


def test_contractions_do_not_become_terms():
    freqs = wordcloud_frequencies(["i'm fine but he isn't safe"])
    assert not any("'" in term for term in freqs)
    assert 'safe' in freqs


def test_digits_and_very_short_tokens_are_dropped():
    # Anything under three letters is noise at word-cloud scale, and bare
    # numbers (years, counts, phone fragments) carry no meaning in the picture.
    freqs = wordcloud_frequencies(['a b ok 12 2026 help'])
    assert set(freqs) == {'help'}


def test_corpus_terms_covers_the_hashtags_actually_in_the_data():
    # A guard: these are the terms measured to dominate the `cleaned` table.
    for term in ('domesticviolence', 'violence', 'violenceagainstwomen'):
        assert term in CORPUS_TERMS
