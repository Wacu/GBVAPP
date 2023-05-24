import streamlit as st
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
#import plotly as plt
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


#def bar_plot(data,)

def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (12, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    
    st.pyplot()
    