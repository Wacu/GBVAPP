import streamlit as st
st.set_page_config(layout='wide',page_title='NLP-WEB-APP',page_icon='🧮')


import pandas as pd
from apps.functions import *
from apps.models import FunctionText2Vec ,FunctionPredictUrgency
#from st_aggrid import  AgGrid
import numpy as np


from apps.functions import *
from apps.cleaning import *
from apps.eda import *
from apps.sentiments import *
from apps.models import *
from apps.gbv_models import load_models




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



tablename3='cleaned'

# Load models

# models.npz is version-independent; see apps/gbv_models.py and
# tools/export_model_params.py for why the joblib pickle was retired.
loaded_models = load_models()
Logistic_Regression=list(loaded_models.keys())[0]
svm=list(loaded_models.keys())[1]


def prediction_confidence(model, predicted, proba, row=0):
    """Probability the model assigned to the class it actually predicted.

    Looks the predicted label up in model.classes_ rather than assuming the
    class value equals its column index. Note that for the SVM, predict()
    (one-vs-one voting) and predict_proba() (Platt coupling) can disagree, so
    this is not always the maximum probability -- it is the confidence in the
    label being displayed, which is what the caller wants.
    """
    cls = predicted[row]
    col = int(np.where(np.asarray(model.classes_) == cls)[0][0])
    return np.round(proba[row][col] * 100, 0)



st.title('GBV Form Detection')
st.markdown('This Section detects the form of GBV in Tweets using the Trained models \
    ')
st.write('\n')

choice=st.sidebar.selectbox('Detect by ',['Text Input', 'Use Cleaned Data'])
  
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
            return 'Economic Violence'
        elif predicted==1:
            return 'Emotional Violence'
        elif predicted==2:
            return 'Physical Violence'
        elif predicted==3:
            return 'Sexual Violence'
        else:
            return 'No violence detected'
    #list(loaded_models.keys())[0]
    if  model_choice == model_options[1]:# Logistic Regression
        model=list(loaded_models.values())[0]
        prediction_choice=predict_function(text,model)
        prediction_label=label_prediction(prediction_choice[0])
        prediction_proba=prediction_confidence(model,prediction_choice[0],prediction_choice[1])
    elif model_choice ==model_options[2]:#svm
        model=list(loaded_models.values())[1]
        prediction_choice=predict_function(text,model)
        prediction_label=label_prediction(prediction_choice[0])
        prediction_proba=prediction_confidence(model,prediction_choice[0],prediction_choice[1])
    else :
        'Select a model of choice'
        model=list(loaded_models.values())[1]
        prediction_choice=predict_function(text,model)
        prediction_label=label_prediction(prediction_choice[0])
        prediction_proba=prediction_confidence(model,prediction_choice[0],prediction_choice[1])
                
    st.subheader('Results')
    st.write('Prediction Labels')
    st.success(prediction_label)
    st.write('Prediction Probability')
    st.info(prediction_proba)

if choice == "Use Cleaned Data":
    data=read_selected_data(tablename3)
    text=[]
    predictions =[]
    with st.form('cleaned'):
        model_options2=['Choose model','Logistic Regression','SVM']
        model_choice2= st.selectbox('Trained  Classify Models Availabe',model_options2)
        submitted_clean= st.form_submit_button('Predict GBV form')
        if submitted_clean:
            def predict_function(text,model):   
                X=FunctionText2Vec(text)
                predicted=model.predict(X)
                predicted_prob=model.predict_proba(X)
                return predicted , predicted_prob                                                                                                                  
        
            def label_prediction(predicted): 
                if predicted==0:   
                    return 'Economic Violence'
                elif predicted==1:
                    return 'Emotional Violence'
                elif predicted==2:
                    return 'Physical Violence'
                elif predicted==3:
                    return 'Sexual Violence'
                else:
                    return 'No violence detected'
                
            if  model_choice2 == model_options2[1]:# Logistic Regression
                model=list(loaded_models.values())[0]
                prediction_choice=predict_function(text,model)
                prediction_label=label_prediction(prediction_choice[0])
                prediction_proba=prediction_confidence(model,prediction_choice[0],prediction_choice[1])
            elif model_choice2 ==model_options2[2]:#svm
                model=list(loaded_models.values())[1]
                prediction_choice=predict_function(text,model)
                prediction_label=label_prediction(prediction_choice[0])
                prediction_proba=prediction_confidence(model,prediction_choice[0],prediction_choice[1])
            else :
                model=list(loaded_models.values())[1]
                prediction_choice=predict_function(text,model)
                prediction_label=label_prediction(prediction_choice[0])
                prediction_proba=prediction_confidence(model,prediction_choice[0],prediction_choice[1])
                        
            st.subheader('Results')
            st.write('Prediction Labels')
            st.success(prediction_label)
            st.write('Prediction Probability')
            st.info(prediction_proba)
            

