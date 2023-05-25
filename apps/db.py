import sqlite3
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

query = "SELECT * FROM users"
tablename='scrape_tweets'

def create_table(df,tablename):

    try:
        conn = sqlite3.connect('tweets.db')
        df.to_sql(tablename, conn, if_exists='append', index=False,dtype='REAL')
        conn.close()  
    except Exception as error:
        print(error)


def read_data(numrows,tablename):

    try:
        query = f"SELECT  * FROM {tablename} LIMIT {numrows}"
        conn = sqlite3.connect('tweets.db')
        df = pd.read_sql_query(query, conn)
        conn.close() 
        return df 
    except Exception as error:
        print(error)

def to_format_our_data_before_store(df):


    df['date_'] = df['date'].dt.strftime('%d/%m/%Y')
    df['date_']= pd.to_datetime(df['date_'])
    df.drop(columns=['date'],inplace=True)

    return df


def read_number_tables():
    conn = sqlite3.connect('tweets.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    list_of_tables =[table[0] for table in tables]
    conn.close()
    return list_of_tables 

print(read_number_tables)






