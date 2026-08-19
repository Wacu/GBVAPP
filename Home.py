import streamlit as st
import pandas as pd
import numpy as np
import plotly as plt
from PIL import Image
from pathlib import Path


# set_page_config must be the first Streamlit call in the script, so it has to
# precede st.header below. The banner in assets/ is 300x168, too wide to read as
# a favicon, hence an emoji icon. layout="wide" matches pages/03.
st.set_page_config(
    page_title="GBV APP",
    page_icon="📊",
    layout="wide",
)

st.header('GBV APP')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """


st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title(" Natural Language Processing")
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
GBVImage = current_dir / "assets" / "GBV.png"
GBVImage = Image.open(GBVImage)
st.image(GBVImage, width=800)
