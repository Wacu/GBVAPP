import streamlit as st
from apps.db import *

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
            st.success(f'The data of {data.shape[0]} has been successful loaded')
            st.table(data.head(2)) 

expander_clean=st.expander('This section View the Selected Data and Go through the Cleaning Process👇')
options_clean=st.selectbox('Select Text  Cleaning Steps',['Text Cleaner'],label_visibility="visible" )

if options_clean == 'Text Cleaner':
    data = cleaning(read_data(int(number_of_rows),tablename))
    
            

    



