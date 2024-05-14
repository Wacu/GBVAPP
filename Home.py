import streamlit as st
import pandas as pd
import numpy as np
import plotly as plt
from PIL import Image
from pathlib import Path


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
current_dir = Path(__file__).parent if "__file__" in locals () else Path.cwd
GBVImage = current_dir / "assets" / "GBV.png"
GBVImage = Image.open(GBVImage)
#st.image(icon,width=800)
