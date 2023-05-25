import streamlit as st
from apps.db import *
from apps.cleaning import *
from apps.sentiments import *
from apps.eda import *
from apps.functions import *

st.set_option('deprecation.showPyplotGlobalUse', False)

tablename='scrape_tweets'

option = st.selectbox('Select data ',['Select one of the below','Choose data from DB', 'Upload Data'],index=0)

if option == 'Select one of the below':
    pass

if option=='Choose data from DB':

    with st.form("my_form"):
        # select_table    =   st.selectbox('Select Table',[read_number_tables])
        number_of_rows  =   st.number_input('Number of Rows')

        submitted =  st.form_submit_button("Submit")
        if submitted: 
            
            data=read_data(int(number_of_rows),tablename)
            st.success(f'The data of {data.shape[0]} has been successful loaded, below is a sample of the data')
            st.table(data.head(2)) 


expander_clean=st.expander('This section View the Selected Data and Go through the Cleaning Process👇')

options_clean=st.selectbox('Select Text  Cleaning Steps',['Steps','Text Cleaner'],label_visibility="visible" )
if options_clean == 'Steps':
    pass
if options_clean == 'Text Cleaner':
    data=read_data(int(number_of_rows),tablename)
    data['clean_text'] = data['text'].apply(text_cleaning)
    st.success('The data has been cleaned,please compare the original and the clean tweet')
    st.table(data[['text','clean_text']].head(5))

# options_sentiment = st.selectbox('Sel')  
#   vader_sentiment         

option_sentiment = st.selectbox('Select a step ',['Select one of the below','Sentiment Analysis','View WordCloud','Topic Modelling'],index=0)

if option_sentiment == 'Select one of the below':
    pass

if option_sentiment=='Sentiment Analysis':

    with st.form("my_sentiment"):

        generate =  st.form_submit_button("Generate")
        if generate: 
            
            data['sentiment'] = data['clean_text'].apply(vader_sentiment)
            st.success('Sentiments have been generated!')
            st.table(data['sentiment'].value_counts(normalize=True)) 
            st.bar_chart(data['sentiment'].value_counts(normalize=True)) 

if option_sentiment == 'View WordCloud':
    with st.form("wordcloud"):
        view = st.form_submit_button("View")
        if view:
            cloud(' '.join(data['clean_text']))
            st.success('The Bigger the font, the more the word was used in the tweets')

if option_sentiment == 'Topic Modelling':
    st.map(data['coordinates'])
    pass


