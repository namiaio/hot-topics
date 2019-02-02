# Hot-topics

An experiment to identify and group similar topics from Finnish newspaper articles by combining Latent Semantic Analysis and HDBSCAN clustering algoritm.

## Dependencies
- **HDBSCAN** - Clustering algorithm
- **Pandas** - Data structuring and analysis
- **Scikit-learn** - Machine learning tools for data mining and data analysis
- **NLTK** - Natural language toolkit
- **HFST** - Helsinki Finite-State Transducer toolkit Python bindings
- **Flask** - Web framework
- **Marshmallow** - Object serialization
- **Bokeh** - Visualization library


## Usage
### Install
Get [compiled](https://github.com/flammie/omorfi/releases) **omorfi.describe.hfst** model or [build it yourself](https://github.com/flammie/omorfi). Then:

`pipenv install`

## Steps

- Get some articles. Used data fields are **title**, **ingress**, **body**, **datetime**

- Load and preprocess data with Pandas

- Give weights to words using TF-IDF (term frequency–inverse document frequency) algoritm. Use Helsinki Finite-State Transducer toolkit with Omorfi lemmatization model for word lemmatization. Stem words using nltk SnowballStemmer.

- Combine TF-IDF with Latent Semantic Analysis for a matrix dimensionality reduction

- Identify natural groupings of the documents using density based document clustering algoritm HDBSCAN
 
- After clustering, topics are created for each cluster using topic modeling technique called Latent Dirichlet Allocation

- Visualize clusters with Bokeh

- Serialize generated result object with Marshmallow

- Use Flask to serve the visualization and JSON result


# Concepts

### Unsupervised learning

Unsupervised Learning is a class of Machine Learning techniques to find the patterns in data. The data given to unsupervised algorithm are not labelled, which means only the input variables are given with no corresponding output variables. In unsupervised learning, the algorithms are left to themselves to discover interesting structures in the data. [Source](https://towardsdatascience.com/unsupervised-learning-with-python-173c51dc7f03)

### Clustering

Clustering is a Machine Learning technique that involves the grouping of data points. Given a set of data points, we can use a clustering algorithm to classify each data point into a specific group. In theory, data points that are in the same group should have similar properties and/or features, while data points in different groups should have highly dissimilar properties and/or features. Clustering is a method of unsupervised learning and is a common technique for statistical data analysis used in many fields. [Source](https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68)

### HDBSCAN

HDBSCAN is a clustering algorithm developed by Campello, Moulavi, and Sander. It extends DBSCAN by converting it into a hierarchical clustering algorithm, and then using a technique to extract a flat clustering based in the stability of clusters. [Source](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

### TF-IDF

TF-IDF, which stands for term frequency — inverse document frequency, is a scoring measure widely used in information retrieval or summarization. TF-IDF is intended to reflect how relevant a term is in a given document. [Source](https://www.kdnuggets.com/2018/08/wtf-tf-idf.html)

### Latent Semantic Analysis (LSA)

Latent Semantic Analysis (LSA) is a theory and method for extracting and representing the contextual-usage meaning of words by statistical computations applied to a large corpus of text. [Source](http://mccormickml.com/2016/03/25/lsa-for-text-classification-tutorial/)

### Latent Dirichlet Allocation

Latent Dirichlet Allocation (LDA) algorithm is an unsupervised learning algorithm that attempts to describe a set of observations as a mixture of distinct categories. LDA is most commonly used to discover a user-specified number of topics shared by documents within a text corpus. [Source](https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html)