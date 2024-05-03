import streamlit as st
from apps.db import *
from apps.cleaning import *
from apps.sentiments import *
from apps.eda import *
from apps.functions import *

st.set_option('deprecation.showPyplotGlobalUse', False)

tablename2='selected'

expander_clean=st.expander('This section View the Selected Data and Go through the Cleaning Process👇')

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

# options_sentiment = st.selectbox('Sel')  
#   vader_sentiment         

option_sentiment = st.selectbox('Select a step ',['Select one of the below','Sentiment Analysis','View WordCloud','Topic Modelling'],index=0)

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
    yearsummary(data['year'])
    cloud(' '.join(data['lemma_nostops']))
    st.success('The Bigger the font, the more the word was used in the tweets')

if option_sentiment == 'Topic Modelling':
    most_freq_words(data['lemma_nostops'])
    bigrams(data['lemma_nostops'])
    trigrams(data['lemma_nostops'])
    #st.map(data)
    #pass
    #most_freq_words(df['lemma_nostops'])


