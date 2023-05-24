import streamlit as st
import pandas as pd
import numpy as np
import plotly as plt


st.header('GBV APP')

# st.set_page_config(
#     page_title="Hello",
#     page_icon="👋",
# )


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """


st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title(" Natural Language Processing")

st.image('https://github.com/Wacu/GBVAPP/blob/5da999594ed361ba5f83510d2919f858f9e4729f/GBV.png',use_column_width=False)