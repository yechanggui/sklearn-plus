#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is used for preprocessing the chinese sentence.  It contains
 JiebaTokenizer class which used for segmenting chinese sentence into words.

"""
from __future__ import unicode_literals
import jieba

from sklearn.base import BaseEstimator, TransformerMixin


class JiebaTokenizer(BaseEstimator, TransformerMixin):
    """Transforms chinese sentence to words by jieba.

    This JiebaTokenizer transforms chinese sentence to words by jieba package.
    You can add user-defined words to fine-tuning the segment result
    when new a instance.

    Parameters
    ----------
    words : array_like, default=None
        user-defined words.

    Attributes
    ----------
    tokenizer : Tokenizer
        a jieba tokenizer object.

    """

    def __init__(self, words=None):
        self.tokenizer = jieba.Tokenizer()

        if words is not None:
            for word in words:
                self.tokenizer.add_word(word)

    def fit(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        """Segemnting chinese sentence into words by jieba.

        Parameters
        ----------
        X : array-like
            Input data that will be transformed.
        """
        return [u' '.join(self.tokenizer.lcut(x)) for x in X]
