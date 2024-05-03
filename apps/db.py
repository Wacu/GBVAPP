import sqlite3
import pandas as pd
import warnings
from datetime import datetime as dt
warnings.filterwarnings('ignore')

query = "SELECT * FROM users"
tablename='unlabelled'
tablename2='selected'


def create_table(df,tablename2):

    try:
        conn = sqlite3.connect('gbv.db')
        df.to_sql(tablename2, conn, if_exists='replace', index=False,dtype='REAL')
        conn.close()  
    except Exception as error:
        print(error)


def read_data(numrows,tablename):

    try:
        query = f"SELECT  * FROM {tablename} LIMIT {numrows}"
        conn = sqlite3.connect('gbv.db')
        df = pd.read_sql_query(query, conn)
        conn.close() 
        return df 
    except Exception as error:
        print(error)

def read_selected_data(tablename2):

    try:
        query = f"SELECT  * FROM {tablename2}"
        conn = sqlite3.connect('gbv.db')
        df = pd.read_sql_query(query, conn)
        conn.close() 
        return df 
    except Exception as error:
        print(error)
'''
def to_format_our_data_before_store(df):
    df['date'] = df['date8'].dt.strftime('%d/%m/%Y')
    df['date_']= pd.to_datetime(df['date_'])
    df.drop(columns=['date'],inplace=True)
    return df
'''


def read_number_tables():
    conn = sqlite3.connect('gbv.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    list_of_tables =[table[0] for table in tables]
    conn.close()
    return list_of_tables 

print(read_number_tables)






