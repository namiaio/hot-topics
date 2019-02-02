# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from hot_topics import Clusters
from hot_topics.helpers import STOP_WORDS, clusterTokenizer

from flask import Flask

app = Flask(__name__)


def load_data():
    df = pd.read_csv("articles.csv")
    
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['content'] = df['title'] + ' ' + df['ingress'] + ' ' + df['body']

    return df


df = load_data()

# Convert a collection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(
    use_idf=True,
    tokenizer=clusterTokenizer,
    stop_words=STOP_WORDS,
    max_features=10000,
    lowercase=True,
    ngram_range=(1, 4)
)

X = vectorizer.fit_transform(df['content'].values)

# Use SVD to perform dimensionality reduction on the tf-idf vectors
lsa = make_pipeline(TruncatedSVD(n_components=300), Normalizer(copy=False))
X_lsa = lsa.fit_transform(X)

#  Generate clusters
topics = Clusters(df, X_lsa)


@app.route('/')
def index():
    return topics.scatter_plot()


@app.route('/result')
def result():
    return topics.to_JSON()


if __name__ == "__main__":
    app.run()
