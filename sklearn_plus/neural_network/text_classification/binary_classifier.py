#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

from ...base import ModelMixin
from .models import XPlusB


class BiClassifier(BaseEstimator, ClassifierMixin, ModelMixin):
    """Biclassifier for contributors to read and try.

    This class define training(fit) and predict process.

    Problem Definition:

    I faked data like this:
        x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
        y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))

    Then I defined a model to do binary classification:
        y = x + b
        prob = sigmoid(y)
        loss = cross_entropy(prob)
        b: the weight to learn.

    Notes:
        1. Every class in this layer (maybe call end to end layer) should inherit
        BaseEstimator, ModelMixin.

        2. At least rewrite fit and predict function.

        3. Call 'super(YourClass, self).__init__()' first in __init__ function.

        4. 'global_step=self.global_step' must added in minimize or maxmize function.
        Otherwise self.global_step will never increase itself.

    """

    def __init__(self, train_loop=100, learning_rate=0.02, every_checkpoint=10, checkpoint_dir=None, summary_dir=None):
        super(BiClassifier, self).__init__()  # call super init method first
        self.train_loop = train_loop
        self.every_checkpoint = every_checkpoint
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir

    def fit(self, X, y=None):
        # init model
        self.model = XPlusB(hyparameters=100)

        # define train step
        # need add global_step
        train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.model.loss,
                                                                                    global_step=self.global_step)

        # define summary dictionary if needed.
        summaries_dict = {
            'loss': self.model.loss,
            'weight': self.model.b,
        }

        # init variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # training loop
        for i in range(self.train_loop):
            # prepare data pipeline
            rand_idx = np.random.choice(100)

            # prepare feed dict
            feed_dict = {
                self.model.input_x: [X[rand_idx]],
                self.model.input_y: [y[rand_idx]]
            }

            # run training ops
            self.sess.run([train_step, self.model.predictions],
                          feed_dict=feed_dict)

            if i % self.every_checkpoint == 0:
                # if summary is needed
                if self.summary_dir:
                    self.summaries(self.summary_dir, feed_dict, summaries_dict)

                # if checkpoint is needed
                if self.checkpoint_dir:
                    self.save(self.checkpoint_dir)

    def predict(self, x):
        feed_dict = {self.model.input_x: [x]}
        prediction_result = self.sess.run(
            self.model.predictions,
            feed_dict)
        return prediction_result
