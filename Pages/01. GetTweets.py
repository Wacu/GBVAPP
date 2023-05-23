import streamlit as st
import configparser
from apps.functions import *
from apps.db import *
import geocoder

#  SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="GBV PREDICTION APP", page_icon=":woman:")

st.write("""
# An App to Predict the Form of Gender Based Violence

**You can enter *text* or fetch data from** *twitter*      
""")

st.sidebar.header("Please Input the below Parameters")

with st.sidebar.form("my_form"):

    location =   st.text_input('Enter the Location')
    radius_location =   st.text_input('Enter the Radius')
    num_tweets = st.number_input('Enter the number of tweets',1,1000,50)

    submitted =  st.form_submit_button("Submit")

if all(param is not None for param in [location, radius_location, num_tweets]):

    latitude   = geocoder.arcgis(location).lat
    longitude  = geocoder.arcgis(location).lng
    data_returned = GetTweet(api,latitude,longitude,radius_location,num_tweets)
    st.table(data_returned.head(10))
else:
    st.info('Input all paramaters')

option = st.selectbox('Choose the mode of Storage',['Choose method','Store in DB', 'Download Csv'],index=0)

tablename='scrape_tweets'


if option == 'Choose method':
    pass
if option == 'Store in DB':
    try:
        data_returned=to_format_our_data_before_store(data_returned)
        create_table(data_returned,tablename)
        st.success('Data has been Stored Successful')
        st.info('Please Proceed to the Next Page to conduct text Cleaning')
    except Exception as err:
        st.info(err)
if option == 'Download Csv':
    download_csv_file(data_returned, 'tweets','Click here to download csv')
    st.markdown('Proceed to the Next Page to conduct text Cleaning')
































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






    

    




