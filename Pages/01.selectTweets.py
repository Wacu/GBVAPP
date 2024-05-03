import streamlit as st
import configparser
from apps.functions import *
from apps.db import *


#  SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
#st.set_page_config(layout="wide", page_title="GBV PREDICTION APP", page_icon=":woman:")

st.write("""
# An App to Predict the Form of Gender Based Violence

**You can enter *text* or fetch data from** *twitter*      
""")
st.subheader("Fetch data")

st.sidebar.header("Please Input the below Parameters")
tablename='unlabelled'
tablename2='selected'
with st.sidebar.form("my_form1"):
    #search_words =   st.text_input('Enter topic')
    #location =   st.text_input('Enter the Location')
    #radius_location =   st.text_input('Enter the Radius')
    num_tweets = st.number_input('Enter the number of tweets',1,100000,50)

    submitted =  st.form_submit_button("Submit")

if all(param is not None for param in [num_tweets]):

    #latitude   = geocoder.arcgis(location).lat
    #longitude  = geocoder.arcgis(location).lng
    option = st.selectbox('Select data ',['Select one of the below','View the data', 'Upload Data'],index=0)

if option == 'Select one of the below':
    pass

if option=='View the data':

    with st.form("my_form2"):
        # select_table    =   st.selectbox('Select Table',[read_number_tables])
        #number_of_rows  =   st.number_input('Number of Rows')

        submitted =  st.form_submit_button("Submit")
        if submitted: 
            
            data=read_data(int(num_tweets),tablename)
            st.success(f'The data of {data.shape[0]} has been successful loaded, below is a sample of the data')
            st.table(data.head(5)) 
            data = create_table(data,tablename2)
        


else:
    st.info('Input all paramaters')

































# tablename='scrape_tweets'
# # col1 , col2   = st.columns(2)


# with col1:
#     store   = st.button('Click to store in the database')
#     if store:
#         try:
#             data_returned=to_format_our_data_before_store(data_returned)
#             create_table(data_returned,tablename)
#             st.success('Data has been Stored Successful')
#         except Exception as err:
#             st.info(err)

# with col2:
    
#     try:
#         download_csv_file(data_returned, 'tweets','Click here to download csv')
#         st.success('Downloaded Succesful')
#     except Exception as err:
#         st.error(err)






    

    




