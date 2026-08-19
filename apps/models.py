import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 
import gensim
import gensim.downloader as api


@st.cache_resource(show_spinner="Loading GloVe word vectors (first run downloads ~66 MB)...")
def load_model(model_name):
    """Load pretrained vectors once per process; cached across reruns and sessions.

    Previously this ran at import time and then copied every vector into a
    400k-entry dict, so each Streamlit process re-downloaded and re-materialised
    the model. The KeyedVectors object already provides keyed vector lookup.
    """
    return api.load(model_name)


def get_glove_vectors():
    return load_model("glove-wiki-gigaword-50")

def FunctionText2Vec(inpTextData):
    vectorizer = CountVectorizer(stop_words='english')
    X=vectorizer.fit_transform(inpTextData)
    #CountVectorizedData=pd.DataFrame(Z.toarray(),columns=vectorizer.get_feature_names_out())
    #X=vectorizer.transform(inpTextData)
    
    CountVecData=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out())
    WordsVocab = CountVecData.columns
    W2Vec_Data=pd.DataFrame()
    glove = get_glove_vectors()
    
    #looping through each row in the data
    for i in range(CountVecData.shape[0]):
        #initialize the sentense with 0's
        Sentence = np.zeros(50)
        for word in WordsVocab[CountVecData.iloc[i, :]>=1]:
            #print word
            if word in glove.key_to_index:
                Sentence = Sentence + glove.get_vector(word)
        #Append the sentences data frame
        Sentencedf=pd.DataFrame([Sentence])
        W2Vec_Data=pd.concat([W2Vec_Data,Sentencedf],ignore_index=True)
    return (W2Vec_Data)


def FunctionPredictUrgency(inpText,model):
    X=FunctionText2Vec(inpText)
    Prediction=model.predict(X)
    return  Prediction 