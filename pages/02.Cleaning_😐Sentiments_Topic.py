import streamlit as st

# set_page_config must precede any Streamlit command, so it goes before the
# `apps` imports below.
st.set_page_config(
    page_title="Clean and explore - GBV APP",
    page_icon="\U0001f4ca",
    layout="wide",
)

import matplotlib.pyplot as plt
from apps.db import *
from apps.cleaning import *
from apps.sentiments import *
from apps.eda import *
from apps.functions import *

st.title("2 - Clean and explore")
st.caption(
    "Normalise the sampled tweets, then score their sentiment and surface the "
    "words and phrases that recur across the corpus."
)

tablename2='selected'
tablename3='cleaned'

options_clean=st.selectbox('Select Text  Cleaning Steps',['Steps','Text Cleaner'],label_visibility="visible" )
if options_clean == 'Steps':
    pass
if options_clean == 'Text Cleaner':
    data=read_selected_data(tablename2)
    data['clean_tweet'] = data['tweet'].apply(text_cleaning).apply(lem)
    stop = stopwords.words('english')
    data['lemma_nostops'] = data['clean_tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    st.success('The data has been cleaned and lematized,please compare the original, clean and lemmatized tweet')
    st.table(data[['tweet','clean_tweet','lemma_nostops']].head(5))
    data = create_table(data,tablename3)

# options_sentiment = st.selectbox('Sel')  
#   vader_sentiment         

option_sentiment = st.selectbox('Select a step ',['Select one of the below','Sentiment Analysis','View WordCloud','Topic Modelling'],index=0)
data=read_selected_data(tablename3)
if option_sentiment == 'Select one of the below':
    pass

if option_sentiment=='Sentiment Analysis':

    with st.form("my_sentiment"):

        generate =  st.form_submit_button("Generate")
        if generate: 
            
            data['sentiment_vader'] = data['lemma_nostops'].apply(lambda x : vader_sentiment_scores(x))
            data['sentiment_blob']= data['lemma_nostops'].apply(lambda x:TextBlob(x).sentiment.polarity)
            #st.table(data['sentiment_vader'].value_counts(normalize=True)) 
            #st.bar_chart(data['sentiment'].value_counts(normalize=True)) 
           
            figure, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
            figure.suptitle('Sentiment Analysis using Textblob vs Vader')

            sns.histplot(ax= axes[0] ,data=data, x='sentiment_vader',kde=True)
            sns.histplot(ax= axes[1],data=data, x='sentiment_blob',kde=True)
            axes[0].set_title('Sentiments by Vader')
            axes[1].set_title('Sentiments by Textblob')
            st.pyplot(plt.gcf(), clear_figure=True)

            # VADER
            st.subheader("Deeper into VADER")
            data['vader_sent'] = data['sentiment_vader'].apply(get_sentiment)
            data.head()
            data['vader_sent'].value_counts(normalize=True)
            p = data['vader_sent'].value_counts(normalize=True)
            p = p.rename("proportion").reset_index()

            sns.barplot(data=p, y='proportion', x='vader_sent',hue='vader_sent')
            plt.title("Sentiment distribution")
            plt.xlabel("Sentiment Score")
            plt.ylabel("Proportion")
            st.pyplot(plt.gcf(), clear_figure=True)
            st.success('Sentiments have been generated!')

if option_sentiment == 'View WordCloud':
    yearsummary(data['year'])

    # The cloud reads the raw tweet, not lemma_nostops: cleaning removes the
    # '@' and '#' markers needed to drop usernames, and leaves plurals as
    # separate terms. See apps/wordcloud_text.py.
    st.caption(
        'Terms the tweets were collected by (domesticviolence, violence, gbv '
        'and similar) are hidden -- they appear in almost every row, so they '
        'describe the search rather than the findings.'
    )
    keep_corpus_terms = st.checkbox('Show the collection terms anyway', value=False)
    typed = st.text_input(
        'Also hide these terms (comma-separated)',
        help='Use this when one term swamps the picture for a particular view.',
    )
    extra = [term.strip() for term in typed.split(',') if term.strip()]

    cloud(
        data['tweet'].tolist(),
        extra_stopwords=extra,
        drop_corpus_terms=not keep_corpus_terms,
    )
    st.success('The Bigger the font, the more the word was used in the tweets')

if option_sentiment == 'Topic Modelling':
    # Raw tweets, for the same reason as the word cloud -- these charts share
    # its filtering now, so the two views agree on what is frequent.
    tweets = data['tweet'].tolist()
    most_freq_words(tweets)
    bigrams(tweets)
    trigrams(tweets)
    #st.map(data)
    #pass
    #most_freq_words(df['lemma_nostops'])


