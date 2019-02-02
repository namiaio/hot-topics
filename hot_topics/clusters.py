# -*- coding: utf-8 -*-

from .helpers import STOP_WORDS, topicTokenizer
from .json_schema import Result

from hdbscan import HDBSCAN
import pandas as pd

from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.palettes import viridis

from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer


class Clusters(object):
    def __init__(self, df, X):
        self.X = X
        self.data = df
        self.hdbscan = HDBSCAN(min_cluster_size=3).fit(X)
        self.labels = self.hdbscan.labels_
        self.clusters = self.init()

        self.fill()

    # Create empty clusters
    def init(self, clusters={}):
        n = self.labels.max()

        for i in range(-1, n+1):
            clusters.update({i: Cluster(id=i)})

        return clusters

    # Fill each cluster with articles
    def fill(self):
        for i, label in enumerate(self.labels):
            cluster = self.clusters[label]

            cluster.articles.append({
                'title': self.data.iloc[i]['title'],
                'month': self.data.iloc[i]['month'],
                'year': self.data.iloc[i]['year'],
                'ingress': self.data.iloc[i]['ingress'],
            })

            if label != -1:
                cluster.create_topics()

    # Create Bokeh plot
    def scatter_plot(self):
        n_labels = self.labels.max()

        X_reduced = TruncatedSVD(n_components=n_labels, random_state=0)
        X_fitted = X_reduced.fit_transform(self.X)
        X_embedded = TSNE().fit_transform(X_fitted)

        labels = [', '.join(self.clusters[n].topics) for n in self.labels]

        df = pd.DataFrame({
            'x': X_embedded[:, 0],
            'y':  X_embedded[:, 1],
            'cluster': self.labels,
            'title': self.data['title'].values,
            'content': self.data['section'].values,
            'topics': labels,
        })

        n_clusters = len(self.clusters)
        colors = viridis(n_clusters) * 2

        plot = figure(
            output_backend="webgl",
            plot_width=1000,
            plot_height=600,
            tools="pan, wheel_zoom, box_zoom, reset, hover, previewsave",
            x_axis_type=None,
            y_axis_type=None,
            min_border=1,
            sizing_mode='scale_width',
        )

        for cluster, group in df.groupby('cluster'):
            if cluster == -1:
                line = '#bdbdbd'
                fill = 'white'
                marker = 'x'
            else:
                line = colors[cluster]
                fill = colors[cluster]
                marker = 'circle'

            plot.scatter(
                x='x',
                y='y',
                source=group,
                size=12,
                line_color=line,
                marker=marker,
                fill_color=fill,
                alpha=0.4
            )

        plot.select(dict(type=HoverTool)).tooltips = {
            "title": "<b>@title</b>",
            "#": "@cluster",
            "topics": "@topics",
        }

        return file_html(plot, CDN)

    # API result endpoint
    def to_JSON(self):
        return Result().dumps(self)


class Cluster(object):
    def __init__(self, id):
        self.id = id
        self.articles = []
        self.topics = []

    # Use article text content to create cluster based topics
    def create_topics(self, n_topics=10):
        corpus = ''

        for article in self.articles:
            corpus = corpus + article['title'] + ' ' + article['ingress']

        vec = CountVectorizer(
            tokenizer=topicTokenizer,
            ngram_range=(1, 2),
            lowercase=False,
            stop_words=STOP_WORDS
        ).fit([corpus])

        bag_of_words = vec.transform([corpus])

        lda = LatentDirichletAllocation(
            n_components=1,
            max_iter=5,
            learning_method='online',
            learning_offset=50.,
            random_state=0
        ).fit(bag_of_words)

        feature_names = vec.get_feature_names()

        topics = []

        for topic_idx, topic in enumerate(lda.components_):
            topics = topics + (
                [feature_names[i] for i in topic.argsort()[:-n_topics - 1:-1]]
            )

        self.topics = topics
