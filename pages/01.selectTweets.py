import streamlit as st

# set_page_config must precede any Streamlit command, so it goes before the
# `apps` imports below.
st.set_page_config(
    page_title="Select data - GBV APP",
    page_icon="\U0001f4ca",
    layout="wide",
)

from apps.functions import *
from apps.db import *

st.title("1 - Select data")
st.caption(
    "Draw a sample of tweets from the stored corpus. The sample is written back "
    "to the database as the working set for the cleaning step."
)

st.sidebar.header("Parameters")
tablename='unlabelled'
tablename2='selected'
with st.sidebar.form("my_form1"):
    num_tweets = st.number_input('Number of tweets to sample',1,100000,50)

    submitted =  st.form_submit_button("Submit")
if submitted:
    option = st.selectbox('Select data ',['Select one of the below','View the data', 'Proceed to Data Cleaning'],index=0)

    if option == 'Select one of the below':
        pass
    if option == 'Proceed to Data Cleaning':
        st.info('Open **02. Cleaning** in the left sidebar to continue.')

    if option=='View the data':

        with st.form("my_form2"):
            submitted =  st.form_submit_button("Submit")
            if submitted:

                data=read_data(int(num_tweets),tablename)
                st.success(f'Loaded {data.shape[0]} rows. A sample is shown below.')
                st.table(data.head(5))
                data = create_table(data,tablename2)
