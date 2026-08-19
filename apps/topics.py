from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from apps.nltk_setup import ensure_nltk_data

# corpora are downloaded on first use; see apps/nltk_setup.py
ensure_nltk_data()
def most_freq_words(corpus):
    top=20
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))

    plt.suptitle("Most frequent words", fontsize=15)
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 1))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(),columns=["Word","Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(string for string in x))
    dtf_bi.set_index("Word").iloc[:top,:].sort_values(by="Freq").plot(kind="barh", title="Most Common words",legend=False).grid(axis='x')
