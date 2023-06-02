import streamlit as st
import pandas as pd
import numpy as np
import plotly as plt
from PIL import Image


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

icon = Image.open("assets\GBV.png")
st.image(icon,width=800)
