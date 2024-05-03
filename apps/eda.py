import streamlit as st
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import nltk
#import plotly as plt
import matplotlib.pyplot as plt
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
    st.pyplot()

def cloud(data,backgroundcolor = 'white', width = 1500, height = 800):
    wordcloud = WordCloud(background_color = backgroundcolor,
                         width = width, height = height,max_words=50,contour_color='steelblue',contour_width=4).generate(data)
    plt.figure(figsize = (12, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    
    st.pyplot()

#Most Frequent words
def most_freq_words(corpus):
    top=10
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    
    plt.suptitle("Most frequent words", fontsize=15)
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 1))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(),columns=["Word","Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(string for string in x))
    dtf_bi.set_index("Word").iloc[:top,:].sort_values(by="Freq").plot(kind="barh", title="Most Common words",legend=False,color='orange').grid(axis='x')
    st.pyplot()

def bigrams(corpus):
    top=10
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    
    plt.suptitle("Bigrams", fontsize=15)
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 2))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(),columns=["Word","Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(string for string in x))
    dtf_bi.set_index("Word").iloc[:top,:].sort_values(by="Freq").plot(kind="barh", title="Bigrams",legend=False,color='grey').grid(axis='x')
    st.pyplot()

def trigrams(corpus):
    top=10
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    
    plt.suptitle("Trigrams", fontsize=15)
    dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, 3))
    dtf_bi = pd.DataFrame(dic_words_freq.most_common(),columns=["Word","Freq"])
    dtf_bi["Word"] = dtf_bi["Word"].apply(lambda x: " ".join(string for string in x))
    dtf_bi.set_index("Word").iloc[:top,:].sort_values(by="Freq").plot(kind="barh", title="Trigrams",legend=False,color='grey').grid(axis='x')
    st.pyplot() 