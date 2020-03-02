import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

def load_data(database_filepath):
    """Load SQLite database for use in a machine learning model.
    
    Args:
        database_filepath: Location of the SQLite database to load
    
    Returns:
        Features to be used in the ML model
        Categories the ML model will try to predict
        Names of the categories for Y
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    #df = pd.read_sql("select * from messages", engine)
    df = pd.read_sql_table("disaster_messages", con=engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """Tokenizes a text into words.
    Args:
        text: The text field to be tokenized
    Returns:
        A list of lemmatized words in lowercase
    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Creates the machine learning model.
    
    Returns:
        Machine learning model
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])

    # parameters = {
        # #'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 1.0),
        # #'vect__max_features': (None, 5000, 10000),
        # 'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [50, 100],
        # #'clf__estimator__min_samples_split': [2, 3, 4]
    # }
    
    parameters =  {
        'clf__estimator__bootstrap': [True, False], 
        'clf__estimator__min_samples_split': [2, 4],
          } 

   

    cv = GridSearchCV(pipeline, param_grid=parameters)
    #cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, cv=3, verbose=5)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates the machine learning model.
    
    Args:
        model: Machine learning model to evaluate
        X_test: Features set for testing
        Y_test: Categories set for testing
        category_names: Category names
    """

    Y_pred = model.predict(X_test)

    for i, column in enumerate(category_names):
        test_array = np.asarray(Y_test[column])
        pred_array = np.asarray(Y_pred[:,i])
        report = classification_report(test_array, pred_array)
        print('{}:\n{}'.format(column, report))

    return None


def save_model(model, model_filepath):
    """Saves the machine learning model as a pickle file.
    Args:
        model: Machine learning model
        model_filepath: File to save to
    """
    pickle_file = open(model_filepath, 'wb')
    pickle.dump(model, pickle_file)
    return None


def main():
    """Creates a machine learning model.
    
    Args (command-line):
        database_filepath: SQLite database containing training data
        model_filepath: File to save model to
    """

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