#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from sklearn.base import BaseEstimator, TransformerMixin

from nltk.tokenize import RegexpTokenizer

class PuncTokenizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.tokenizer = RegexpTokenizer(ur'\w+')

    def fit(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):

        new_X = []

        for x in X:
            tmp = x
            if isinstance(tmp, str):
                tmp = unicode(tmp, 'utf8')
            tmp = u' '.join(self.tokenizer.tokenize(tmp))
            new_X.append(tmp.strip())
        return new_X
