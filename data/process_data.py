import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from collections import Counter


def load_data(messages_filepath, categories_filepath):
    """
    Loading dataframe from filepaths
    
    INPUT
    messages_filepath -- str, link to file
    categories_filepath -- str, link to file
    
    OUTPUT
    df - pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id',how='inner')

    return df



def clean_data(df):
    """
    Pre-processing the Datafram for modelling
    
    INPUT
    df -- type pandas DataFrame
    
    OUTPUT
    df -- cleaned pandas DataFrame
    """
    categories = df.categories.str.split(pat=';', n=-1, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1])
        categories[column] = categories[column].astype(int)
    categories.related = categories.related.replace(2,0)
    categories.drop('child_alone',axis=1,inplace=True)
    df.drop(['categories','original'],axis=1,inplace=True)
    categories['id'] = df['id'].copy()
    df = df.merge(categories,on='id',how='inner')
    df = df[df.duplicated()==False]

    return df

def save_data(df, database_filename):
    """Saving the DataFrame (df) to a database path"""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql( 
        'msg',
        con=engine, 
        index=False,
        if_exists='replace'
        )  


def main():
    """Running main functions"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
