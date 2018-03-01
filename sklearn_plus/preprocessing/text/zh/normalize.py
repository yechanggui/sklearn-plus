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
            if isinstance(tmp, str):
                tmp = unicode(tmp, 'utf8')
            tmp = re.sub(ur'^http.*$', u' ', tmp)
            tmp = re.sub(ur'[ 、，。？：（）【】〜！/\@\-\",:<>~“”\'\(\)\[\]⋯?\$\!\\\.\_\^]', u' ', tmp)
            tmp = re.sub(ur'^[a-zA-Z0-9\.\*\+\-\_]+$', u' ', tmp)
            new_X.append(tmp.strip())
        return new_X
