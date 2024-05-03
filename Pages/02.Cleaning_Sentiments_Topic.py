import streamlit as st
from apps.db import *
from apps.cleaning import *
from apps.sentiments import *
from apps.eda import *
from apps.functions import *

st.set_option('deprecation.showPyplotGlobalUse', False)

tablename='unlabelled'
num_tweets=50

expander_clean=st.expander('This section View the Selected Data and Go through the Cleaning Process👇')

options_clean=st.selectbox('Select Text  Cleaning Steps',['Steps','Text Cleaner'],label_visibility="visible" )
if options_clean == 'Steps':
    pass
if options_clean == 'Text Cleaner':
    data=read_data(int(num_tweets),tablename)
    data['clean_tweet'] = data['tweet'].apply(text_cleaning)
    st.success('The data has been cleaned,please compare the original and the clean tweet')
    st.table(data[['tweet','clean_tweet']].head(5))

# options_sentiment = st.selectbox('Sel')  
#   vader_sentiment         

option_sentiment = st.selectbox('Select a step ',['Select one of the below','Sentiment Analysis','View WordCloud','Topic Modelling'],index=0)

if option_sentiment == 'Select one of the below':
    pass

if option_sentiment=='Sentiment Analysis':

    with st.form("my_sentiment"):

        generate =  st.form_submit_button("Generate")
        if generate: 
            
            data['sentiment_vader'] = data['clean_tweet'].apply(lambda x : vader_sentiment_scores(x))
            data['sentiment_blob']= data['clean_tweet'].apply(lambda x:TextBlob(x).sentiment.polarity)
            #st.table(data['sentiment_vader'].value_counts(normalize=True)) 
            #st.bar_chart(data['sentiment'].value_counts(normalize=True)) 
           
            figure, axes = plt.subplots(1, 2, sharex=True, figsize=(15,5))
            figure.suptitle('Sentiment Analysis using Textblob vs Vader')

            sns.histplot(ax= axes[0] ,data=data, x='sentiment_vader',kde=True)
            sns.histplot(ax= axes[1],data=data, x='sentiment_blob',kde=True)
            axes[0].set_title('Sentiments by Vader')
            axes[1].set_title('Sentiments by Textblob')
            st.pyplot()

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
            st.pyplot()
            st.success('Sentiments have been generated!')

if option_sentiment == 'View WordCloud':
    with st.form("wordcloud"):
        view = st.form_submit_button("View")
        if view:
            yearsummary(data['year'])
            cloud(' '.join(data['clean_tweet']))
            st.success('The Bigger the font, the more the word was used in the tweets')

if option_sentiment == 'Topic Modelling':
    most_freq_words(data['clean_tweet'])
    bigrams(data['clean_tweet'])
    trigrams(data['clean_tweet'])
    #st.map(data)
    #pass


