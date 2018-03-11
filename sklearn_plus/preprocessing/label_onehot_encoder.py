#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class LabelOneHotEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X):

        # Step 1, from label to int
        self.label_encoder = LabelEncoder()
        X_label = self.label_encoder.fit_transform(X)

        # Step 2, reshape for onehot encoder
        X_label_reshape = X_label.reshape(-1, 1)

        # Step 3, onehot
        self.onehot_encoder = OneHotEncoder(sparse=False)
        self.onehot_encoder.fit(X_label_reshape)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        return self.onehot_encoder.transform(self.label_encoder.transform(X).reshape(-1, 1))

    def inverse_transform(self, X):
        return self.label_encoder.inverse_transform(np.argmax(X, axis=1))
