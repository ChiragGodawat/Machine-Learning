"""
Topic Extraction using Non Negative Matrix Factorization (NMF)
or Latent Semantic Analysis (LSA) also known as Singular Value Decomposition (SVD)

These are the decomposition techniques that reduce the data to a given number of components
"""

import pandas as pd
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition

import re
import string

from src import config


def clean_text(t: str):
    t = t.split()
    t = " ".join(t)
    t = re.sub(f'[{re.escape(string.punctuation)}]', '', t)

    return t


if __name__ == "__main__":
    corpus = pd.read_csv(config.IMDB_DATA, nrows=10000)
    corpus["review"] = corpus.review.apply(clean_text)
    corpus = corpus.review.values

    tfv = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
    tfv.fit(corpus)
    corpus_transformed = tfv.transform(corpus)

    svd = decomposition.TruncatedSVD(n_components=10)
    corpus_svd = svd.fit(corpus_transformed)

    for sample_index in range(10):
        feature_scores = dict(
            zip(
                tfv.get_feature_names(),
                corpus_svd.components_[sample_index]
            )
        )

        N = 5
        print(sorted(feature_scores, key=feature_scores.get, reverse=True)[:N])
