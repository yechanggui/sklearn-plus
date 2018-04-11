#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from sklearn.base import BaseEstimator, TransformerMixin


class Normalizer(BaseEstimator, TransformerMixin):

    def fit(self, X):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        new_X = []

        for x in X:
            tmp = x.strip().lower()
            try:
                tmp = tmp.decode('utf-8')
            except UnicodeEncodeError:
                tmp = tmp
            except AttributeError:
                tmp = tmp
            tmp = re.sub('^http.*$', u' ', tmp)
            tmp = re.sub('[ 、，。？：（）【】〜！/@\-\",:<>~\'()\[\]⋯?$!._^]', u' ', tmp)
            tmp = re.sub(r'^[a-zA-Z0-9.*+\-_]+$', u' ', tmp)
            new_X.append(tmp.strip())
        return new_X




