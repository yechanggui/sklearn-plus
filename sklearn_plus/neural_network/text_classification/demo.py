#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

from ...base import ModelMixin
from .models.demo_model import Demo


class DemoClassifier(BaseEstimator, ClassifierMixin, ModelMixin):
    def __init__(self, checkpoint_dir=None, summary_dir=None):
        super(DemoClassifier, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir

    def fit(self, X, y=None):
        self.model = Demo(1, 100)
        # define train step
        train_step = tf.train.GradientDescentOptimizer(0.02).minimize(self.model.loss,
                                                                      global_step=self.global_step)  # need add global_step

        # define summary dictionary
        summaries_dict = {
            'loss': self.model.loss,
            'weight': self.model.A,
        }

        # init variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # training loop
        for i in range(100):
            # prepare data pipeline
            rand_idx = np.random.choice(100)

            # prepare feed dict
            feed_dict = {
                self.model.x_data: [X[rand_idx]],
                self.model.y_target: [y[rand_idx]]
            }

            # run training ops
            self.sess.run([train_step],
                          feed_dict=feed_dict)

            if i % 25 == 0:
                # if summary is needed
                if self.summary_dir:
                    self.summaries(self.summary_dir, feed_dict, summaries_dict)

                # if checkpoint is needed
                if self.checkpoint_dir:
                    self.save(self.checkpoint_dir)

    def predict(self, x):
        feed_dict = {self.model.x_data: [x]}
        prediction_result = self.sess.run(
            self.model.predictions,
            feed_dict)
        return prediction_result
