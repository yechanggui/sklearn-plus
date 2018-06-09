#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn_plus.utils.data_helpers import batch_iter

import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn

from sklearn_plus.utils import const

from sklearn import preprocessing

class Estimator(BaseEstimator, ClassifierMixin):

    def __init__(self, model_dir, model_fn, params):

        self.model_dir=model_dir
        self.model_fn=model_fn
        self.params=params

        self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.params.max_document_length)
        self.label_processor = preprocessing.LabelEncoder()
    def preprocess_x(self, X):
        return np.array(list(self.vocab_processor.transform(X)))
    def preprocess_y(self, y):
        return self.label_processor.transform(y)
    def postprocess_y(self, y):
        return self.label_processor.inverse_transform(y)

    @classmethod
    def construct_input_fn(cls, word_ids, y=None, batch_size=128, num_epochs=1, shuffle=None):
        return tf.estimator.inputs.numpy_input_fn(
                x={const.word_ids: word_ids},
                y=y,
                batch_size=batch_size,
                num_epochs=num_epochs,
                shuffle=shuffle)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):

        self.vocab_processor.fit(X_train)
        self.params.add_hparam('vocab_size', len(self.vocab_processor.vocabulary_))

        self.label_processor.fit(y_train)
        self.params.add_hparam('class_num', len(self.label_processor.classes_))

        self.classifier = tf.estimator.Estimator(
                model_dir=self.model_dir,
                model_fn=self.model_fn,
                params=self.params)

        train_input_fn = self.construct_input_fn(
                word_ids=self.preprocess_x(X_train),
                y=self.preprocess_y(y_train),
                batch_size=self.classifier.params.batch_size,
                num_epochs=self.classifier.params.num_epochs,
                shuffle=True)
        self.classifier.train(train_input_fn)
        if X_valid is not None and y_valid is not None:
            eval_input_fn = self.construct_input_fn(
                word_ids=self.preprocess_x(X_valid),
                y=self.preprocess_y(y_valid),
                shuffle=True)
            self.classifier.evaluate(eval_input_fn)

    def predict(self, X):
        predict_input_fn = self.construct_input_fn(
            word_ids=self.preprocess_x(X),
            shuffle=False)

        return [self.postprocess_y(p['class']) for p in self.classifier.predict(predict_input_fn)]
#        return [self.postprocess_y(p['class'].argsort()[0-top_n:][::-1]) for p in self.classifier.predict(predict_input_fn)]
#        top_n_idx = y_predict.argsort()[0-top_n:][::-1]
#        classes = self.postprocessor.inverse_transform(topn_idx)
#        return [p['prob'][1] for p in self.classifier.predict(predict_input_fn)]

    def predict_top_n(self, X, n=1):
        predict_input_fn = self.construct_input_fn(
            word_ids=self.preprocess_x(X),
            shuffle=False)

        response = []
        for p in self.classifier.predict(predict_input_fn):
            top_n_idx = p['prob'].argsort()[0-n:][::-1]
            classes = self.postprocess_y(top_n_idx)
            probs = p['prob'][top_n_idx]

            categories = []

            for i in xrange(n):
                item = {}
                item[u'name'] = unicode(classes[i], 'utf8')
                item[u'confidence'] = probs[i]
                categories.append(item)

            response.append({u'categories' : categories})
        return response

