import streamlit as st
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import nltk
#import plotly as plt
import matplotlib.pyplot as plt
from apps.nltk_setup import ensure_nltk_data
from apps.wordcloud_text import top_ngrams, wordcloud_frequencies

# corpora are downloaded on first use; see apps/nltk_setup.py
ensure_nltk_data()
plt.style.use('fivethirtyeight')


#def bar_plot(data,)
def yearsummary(y):
    plt.figure(figsize=(8, 6))
    y.value_counts(normalize=True).sort_index().plot(kind='barh', color='skyblue')
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.title('Proportion of Data Over the Years')
    plt.xticks(rotation=0)
    plt.grid(False)
    st.pyplot(plt.gcf(), clear_figure=True)

def cloud(texts, backgroundcolor='white', width=1600, height=900,
          max_words=55, extra_stopwords=(), drop_corpus_terms=True):
    """Draw a word cloud from raw tweets.

    Takes the tweets themselves, not a pre-joined string of `lemma_nostops`.
    See `apps/wordcloud_text.py` for why: by the time cleaning has run, the
    hashtag and username markers are gone and plurals have not been collapsed.

    The previous version passed `contour_color` and `contour_width` without a
    `mask`. `WordCloud._draw_contour` returns the image untouched when `mask`
    is None, so those two arguments drew nothing at all.
    """
    if isinstance(texts, str):
        texts = [texts]

    frequencies = wordcloud_frequencies(
        texts,
        extra_stopwords=extra_stopwords,
        drop_corpus_terms=drop_corpus_terms,
    )
    if not frequencies:
        st.info('No terms left to draw once filtering was applied.')
        return

    wordcloud = WordCloud(
        background_color=backgroundcolor,
        width=width, height=height,
        max_words=max_words,
        # Counting happens in wordcloud_text so the cloud and the frequency
        # charts agree; WordCloud's own tokeniser would filter differently.
        # collocations would also re-add bigrams of words already counted.
        collocations=False,
        colormap='cividis',
        prefer_horizontal=0.95,
        # Scale font size with frequency rather than with rank, so a term that
        # is genuinely twice as common looks twice as big.
        relative_scaling=0.5,
        min_font_size=12,
        # Without a margin the glyphs pack until letters touch, which is what
        # made the picture hard to read even once the terms were right.
        margin=6,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(12, 7))
    # Without interpolation the upscaled bitmap renders with visibly jagged
    # letter edges.
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    # The fivethirtyeight style paints a grey figure background, which framed
    # the white cloud in a grey border.
    fig.patch.set_facecolor(backgroundcolor)
    fig.tight_layout(pad=0)

    st.pyplot(fig, clear_figure=True)

def _ngram_chart(texts, n, title, color, top=10, extra_stopwords=(),
                 drop_corpus_terms=True):
    """Draw the top n-grams as a horizontal bar chart.

    These charts used to tokenise `lemma_nostops` themselves, so they filtered
    differently from the word cloud and the two could disagree about what was
    frequent. They now share `apps/wordcloud_text`, which also means the
    collection terms are dropped here too -- previously `domesticviolence` was
    the top word in every one of these charts, which is a fact about the search
    rather than the data.
    """
    pairs = top_ngrams(texts, n=n, top=top, extra_stopwords=extra_stopwords,
                       drop_corpus_terms=drop_corpus_terms)
    if not pairs:
        st.info('Not enough text to chart {}.'.format(title.lower()))
        return

    frame = pd.DataFrame(pairs, columns=['Word', 'Freq']).sort_values('Freq')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(frame['Word'], frame['Freq'], color=color)
    ax.set_title(title)
    ax.set_xlabel('Occurrences')
    ax.grid(axis='x')
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


#Most Frequent words
def most_freq_words(texts, **kwargs):
    _ngram_chart(texts, 1, 'Most common words', 'orange', **kwargs)


def bigrams(texts, **kwargs):
    _ngram_chart(texts, 2, 'Bigrams', 'grey', **kwargs)


def trigrams(texts, **kwargs):
    _ngram_chart(texts, 3, 'Trigrams', 'grey', **kwargs) 