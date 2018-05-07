#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from tensorflow.contrib import learn
from sklearn.base import BaseEstimator, TransformerMixin


class TextToBagVec(BaseEstimator, TransformerMixin):
    def __init__(self, max_length=None, min_frequency=1):
        self.max_length = max_length
        self.min_frequency = min_frequency - 1

    def fit(self, X):
        if self.max_length is None:
            self.max_length = max([len(x.split(' ')) for x in X])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.max_length,
                                                                       min_frequency=self.min_frequency)
        self.vocab_processor.fit(X)
        return self.vocab_processor

    def fit_transform(self, X, y=None):
        self.fit(X)
        return np.array(list(self.vocab_processor.transform(X)))

    def transform(self, X, y=None):
        return np.array(list(self.vocab_processor.transform(X)))
