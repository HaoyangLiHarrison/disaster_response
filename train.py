#!/usr/bin/env python
# coding: utf-8

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


engine = create_engine('sqlite:///disaster_messages.db')
df = pd.read_sql_table(table_name='messages_categories',con=engine)

X = df.message
y = df.iloc[:,3:]

def tokenize(text):
    words = word_tokenize(text)
    return words

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
improved_pipeline.fit(X_train, y_train)
#y_pred_improved = improved_pipeline.predict(X_test)
#print(classification_report(y_pred_improved,y_test,target_names=y_test.columns))

joblib.dump(improved_pipeline, 'model.pkl')