import json
import plotly
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download('stopwords')

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # chart 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # chart 2 
    df_cat = df.drop(['message', 'original', 'genre'], axis=1)
    cat_names = [col for col in df_cat.columns if col != 'id']
    df_cat = pd.melt(df_cat, id_vars=['id'], value_vars=cat_names)
    category_totals = df_cat.groupby('variable')['value'].sum()
    category_totals.sort_values(ascending=True, inplace=True)
    x_vals = category_totals.values.tolist()
    y_vals = category_totals.index.tolist()
    
    # chart 3 
    text = ' '.join(df['message']).lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = pd.Series(words)
    words = words[~words.isin(stopwords.words("english"))]
    frequent_words=words.value_counts()[:10]
  
    frequent_word_names = list(frequent_words.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'chart 1: Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_vals,
                    y=y_vals,
                    orientation='h',
                    text=y_vals,
                    textposition='auto'
                )
            ],

            'layout': {
                'title': 'chart 2: Distribution of Message Categories',
                'yaxis': {
                    'showticklabels': False
                },
                'xaxis': {
                    'title': "Messages"
                },
                'autosize': False,
                'width': 500,
                'height': 1000,
                'showlegend': False
            }
        },
        {
            'data': [
                Bar(
                    x=frequent_word_names,
                    y=frequent_words
                )],
            'layout': {
                'title': 'chart 3: Most Frequent Words top 10',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()