import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
import textblob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


#Valence Aware Dictionary sEntiment Reasoner - best for SM language
def vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    vader_score = score['compound'] # compound shows the overall sentiment
    if vader_score > 0:
        return 'Positive'
    elif vader_score < 0:
        return 'Negative'
    else:
        return 'Neutral'

#Using Textblob 
# def blob_sentiment(text):
#     blob_score = TextBlob(text).sentiment.polarity
#     return blob_score