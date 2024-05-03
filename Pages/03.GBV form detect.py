import streamlit as st
st.set_page_config(layout='wide',page_title='NLP-WEB-APP',page_icon='🧮')


import pandas as pd
from apps.functions import *
from apps.models import FunctionText2Vec ,FunctionPredictUrgency
#from st_aggrid import  AgGrid
import joblib
import numpy as np


from apps.functions import *
from apps.cleaning import *
from apps.eda import *
from apps.sentiments import *




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)





# Load models

loaded_models= joblib.load('models.joblib')

Logistic_Regression=loaded_models[0].name
svm=loaded_models[1].name


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('GBV Form Detection')
st.markdown('This Section detects the form of GBV in Tweets using the Trained models \
    ')
st.write('\n')

choice=st.sidebar.selectbox('Detect by ',['Text Input', 'Cleaned Data'])
  
if choice == 'Text Input':
    with st.form('my_form'):  
        txt=st.text_area(label='Input a text',value=' Delete me and Paste your comments here and I will predict for you the form of GBV')
        model_options=['Choose model','Logistic Regression','SVM']
        model_choice= st.selectbox('Trained  Classify Models Availabe',model_options)
        submitted= st.form_submit_button('Predict')
        text=[]
        text.append(txt)
    
    
    def predict_function(text,model):   
        X=FunctionText2Vec(text)
        predicted=model.predict(X)
        predicted_prob=model.predict_proba(X)
        return predicted , predicted_prob
        
    def label_prediction(predicted): 
        
        if predicted==0:   
            return 'No Hate'
        else:
            return 'Hate'

    if  model_choice == model_options[1]:# Logistic Regression
        prediction_choice=predict_function(text,loaded_models[0])
        prediction_label=label_prediction(prediction_choice[0])
        prediction_proba=np.round(prediction_choice[1][0][0] *100,0)    
    elif model_choice ==model_options[2]:#svm
        prediction_choice=predict_function(text,loaded_models[10])
        prediction_label=label_prediction(prediction_choice[0])
        prediction_proba=np.round(prediction_choice[1][0][1] *100,0)   
    else :
        prediction_choice=predict_function(text,loaded_models[6])
        prediction_label=label_prediction(prediction_choice[0])
        prediction_proba=np.round(prediction_choice[1][0][1] *100,0)
                
    st.subheader('Results')
    st.write('Prediction Labels')
    st.success(prediction_label)
    st.write('Prediction Probability')
    st.info(prediction_proba)


