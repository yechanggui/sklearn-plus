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
            tmp = re.sub('^http.*$', u' ', tmp)
            tmp = re.sub('[ 、，。？：（）【】〜！/@\-\",:<>~\'()\[\]⋯?$!._^]', u' ', tmp)
            tmp = re.sub(r'^[a-zA-Z0-9.*+\-_]+$', u' ', tmp)
            new_X.append(tmp.strip())
        return new_X


test_str = '详情查看：http://www.ruyi.ai/，或者关注微信公众号“艾如意宝宝”获取更多信息' # utf-8
print(list(test_str))
print(type(test_str).__mro__)
print(list(test_str.decode('utf-8'))) # utf-8 to unicode
test_str2 = u'详情查看：http://www.ruyi.ai/，或者关注微信公众号“艾如意宝宝”获取更多信息' # unicode
print(list(test_str2))

normalizer = Normalizer()
normalizer.transform(
            [u'详情查看：http://www.ruyi.ai/，或者关注微信公众号“艾如意宝宝”获取更多信息']
        )



