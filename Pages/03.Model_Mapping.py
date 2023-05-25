import streamlit as st
from apps.functions import *

st.title("GBV Form Prediction App")
st.markdown("This section of the App is used to detect the form of GBV in tweets using trained models")
st.write('\n')

option_detect= st.selectbox('Detect using ',['Input Text', 'Proceed with Tweets'])



#Mapping
