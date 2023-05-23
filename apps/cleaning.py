import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.stem import PorterStemmer
import re
import string
from string import punctuation
stop_words = set(nltk.corpus.stopwords.words('english'))
from apps.db import *


def text_cleaning(text):    
    def pad_str(s):
        return ' '+s+' '
    
    # Empty question
    if type(text) != str or text=='':
        return ''
    # Clean the text
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub("[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub("\w*\d\w*", "", text)


    text = re.sub("\'s", " ", text) 
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text)
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text)
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text)
    text = re.sub("\'d", " would ", text)
    text = re.sub("\'ll", " will ", text)
    text = re.sub("e\.g\.", " eg ", text, flags=re.IGNORECASE)
    text = re.sub("(\d+)(kK)", " \g<1>000 ", text)
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)
    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    text = re.sub("[c-fC-F]\:\/", " disk ", text)
    
    # remove comma between numbers, i.e. 15,000 -> 15000
    text = re.sub('(?<=[0-9])\,(?=[0-9])', "", text)
    
    def pad_number(pattern):
        matched_string = pattern.group(0)
        return pad_str(matched_string)
    text = re.sub('[0-9]+', pad_number, text)
    
    # add padding to punctuations and special chars, we still need them later
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub('amp', "", text)
    
    def pad_pattern(pattern):
       matched_string = pattern.group(0)
       return pad_str(matched_string)
    text = re.sub('[\!\?\@\^\+\*\/\,\~\|\`\=\:\;\.\#\\\]', pad_pattern, text) 
        
    
    text = [c for c in text if c not in string.punctuation]
    text = [c for c in text if c != '\n'] # Removing newline
    text = ''.join(text)
    text = text.split('https')[0] # Remove links
    
    # replace the float numbers with a random number, it will be parsed as number afterward, and also been replaced with word "number"
    text = re.sub('[0-9]+\.[0-9]+', " 87 ", text)
    
    # Search for all non-letters # Replace all non-letters with spaces
    text = re.sub("[^a-zA-Z]",  
                          " ",          
                          str(text))
    
    
    #stem words using nltk and remove stopwords
    stemmer = PorterStemmer()
    words = text.split()
    nostops_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in nostops_words]
    text = ' '.join(stemmed_words)
    
    #stem words using spacy - took so long
#     stemmerspacy=spacy.load("en_core_web_sm")
#     doc = stemmerspacy(text)
#     spacywords = [token.lemma_ for token in doc]
    #spacywords = [word for word in words if word not in stop_words]
    #text = ''.join(spacywords)
    
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation]).lower()
       # Return a list of words
    return text