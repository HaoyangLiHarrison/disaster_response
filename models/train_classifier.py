import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk import word_tokenize
import time
import re
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
pd.set_option('mode.chained_assignment', None)
nltk.download('punkt')

def load_data(database_filepath):
    """
    Loading the pre-processed DataFrame from the db path.
    Specifying the training data namely X and y as well as 
    the categorys of labels for model evaluation. 
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table_name='msg',con=engine)
    X = df.message
    y = df.iloc[:,3:]
    category_names = y.columns

    return X,y,category_names


def tokenize(text):
    '''
    Tokenlizing the inpute Series of text as words.
    '''
    words = word_tokenize(text)
    return words


def build_model():
    '''
    Constructing the ML pipline for trianing and 
    classification.
    '''
    improved_pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize,stop_words='english')),
        (
            'clf', MultiOutputClassifier(
                AdaBoostClassifier(
                    random_state = 42,
                    learning_rate = 0.3,
                    n_estimators = 200
                )
            )
        ),
    ])
    return improved_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluating the model performance. 
    '''
    Y_pred_improved = model.predict(X_test)
    print(classification_report(Y_pred_improved,Y_test,target_names=category_names))


def save_model(model, model_filepath):
    """Saving the pipeline as pickle file"""
    joblib.dump(model, model_filepath)


def main():
    """Loading the data, running the model and saving the model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
