#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba

from sklearn.base import BaseEstimator, TransformerMixin

class JiebaTokenizer(BaseEstimator, TransformerMixin):

    def __init__(self, words=None):
        self.tokenizer = jieba.Tokenizer()

        if words is not None:
            for word in words:
                if isinstance(word, str):
                    word = unicode(word, 'utf8')
                self.tokenizer.add_word(word)

    def fit(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return [u' '.join(self.tokenizer.lcut(x)) for x in X]
