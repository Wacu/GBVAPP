import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
import gensim
import gensim.downloader as api


def load_model(model_name):
    model = api.load(model_name)
    return model

model=load_model("glove-wiki-gigaword-50")


GloveWordVectors = {}
for word in model.key_to_index.keys():
    GloveWordVectors[word] = model.get_vector(word)
  

# if 'dictionary' not in st.session_state:
#     st.session_state.dictionary=GloveWordVectors

def FunctionText2Vec(inpTextData):
    
    # if 'dictionary' not in st.session_state:
    #     st.session_state.dictionary=GloveWordVectors
    
    
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(inpTextData)
    CountVecData=pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()) 
    WordsVocab=CountVecData.columns
    W2Vec_Data=pd.DataFrame()
    for i in range(CountVecData.shape[0]):
        Sentence = np.zeros(50)
        for word in WordsVocab[CountVecData.iloc[i, :]>=1]:
            if word in GloveWordVectors.keys():    
                Sentence=Sentence+GloveWordVectors[word]
        W2Vec_Data=W2Vec_Data.append(pd.DataFrame([Sentence]))
    return W2Vec_Data


def FunctionPredictUrgency(inpText,model):
    X=FunctionText2Vec(inpText)
    Prediction=model.predict(X)
    return  Prediction 