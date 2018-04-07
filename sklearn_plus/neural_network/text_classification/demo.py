#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import sys

sys.path.append('/Users/liuxiaoan/PycharmProjects/sklearn-plus/sklearn_plus')
# print sys.path
import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn_plus.base import ModelMixin

from models.demo_model import Demo


class DemoClassifier(BaseEstimator, ClassifierMixin, ModelMixin):
    def __init__(self):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        self.sess = tf.Session()
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.model = Demo()

    def fit(self, X, y=None):
        train_step = tf.train.GradientDescentOptimizer(0.02).minimize(self.model.loss,
                                                                      global_step=self.global_step)  # need add global_step

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for i in range(100):
            rand_idx = np.random.choice(100)
            self.sess.run([train_step],
                          feed_dict={self.model.x_data: [X[rand_idx]], self.model.y_target: [y[rand_idx]]})
            # if i % 25 == 0:
                # print 'Step %d, A = %.4f' % (i, self.sess.run(self.model.A))
                # print 'Loss = %f' % self.sess.run(self.model.loss,
                #                                   feed_dict={self.model.x_data: [x_vals[rand_idx]],
                #                                              self.model.y_target: [y_vals[rand_idx]]})
                # self.save('/Users/liuxiaoan/Downloads/sklearn_plus_test')

    def predict(self, x):
        feed_dict = {self.model.x_data: [x]}
        prediction_result = self.sess.run(
            self.model.predictions,
            feed_dict)
        return prediction_result


if __name__ == '__main__':
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(-1., 50), np.repeat(1., 50)))
    print 'data is ready...'

    clf = DemoClassifier()
    clf.fit(x_vals, y_vals)
    print clf.save('/Users/liuxiaoan/Downloads/sklearn_plus_test')
    print 'saved weight: %f' % clf.sess.run(clf.model.A)

    del clf
    print 'del model ...'
    clf = DemoClassifier()
    print clf.load('/Users/liuxiaoan/Downloads/sklearn_plus_test')
    print 'loaded weight: %f' % clf.sess.run(clf.model.A)
    print clf.predict(-2)
    print clf.predict(4)
