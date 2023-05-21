import tweepy
import pandas  as pd
import streamlit as st

#####################     Define the Secret #################################

consumer_key         ='16b7VCeoiTxSuNNRqjGPBSIws'
consumer_secret      ='ytMKOh0hkRK7fOfgXpZXmoySXp1sekWU6lpspmpGHV8YHv1Ylg'
access_token         ='1389549456916369408-1AxDbFYZ6dDzaIkpTPsGPKaURZMagz'
access_token_secret  ='lmnidkW9ZsP5zLAnIrWPzIVpiQ3AhsFQsM9sVWSL5tGMd'

##################### Scrape Function ########################
search_words = "#violence OR #genderbasedviolence OR #Activismagainstgbv OR #murder OR #gbv OR #SexualAbuse OR #DomesticAbuse OR #ViolenceAgainstWomen OR #harassment OR #femicide OR #rape OR #domesticviolence OR #sexualassault"


# Create the API object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


def GetTweet(api,latitude,longitude,radius,num_of_tweets):
 
    tweets = tweepy.Cursor(api.search_tweets
                           ,q=search_words,
                            geocode = f'{latitude},{longitude},{radius}km',
                            lang="en",
                            ).items(int(num_of_tweets))

    # Iterate and print tweets
    data = [[ tweet.created_at ,tweet.user.screen_name, tweet.user.location, tweet.text] for tweet in tweets]

    df = pd.DataFrame(data=data, columns=['date','user', 'location', 'text'])

    return df


def download_csv_file(df,name,label_name):
    csv=df.to_csv().encode('utf-8')
    st.download_button(
    label=label_name,
    data=csv,
    file_name=f'{name}.csv',
    mime='text/csv',)

