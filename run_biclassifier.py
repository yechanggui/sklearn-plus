#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn_plus.neural_network.text_classification.binary_classifier import BiClassifier

if __name__ == "__main__":
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
    print('data is ready...\n')

    clf = BiClassifier(summary_dir='/tmp/biclassifier_model',
                       checkpoint_dir='/tmp/biclassifier_model')
    clf.fit(x_vals, y_vals)
    print('save path: %s' % clf.save('/tmp/biclassifier_model'))
    print('saved weight: %f\n' % clf.sess.run(clf.model.b))

    del clf
    print('del model ...')
    clf = BiClassifier()
    clf.load('/tmp/biclassifier_model')
    print('loaded weight: %f' % clf.sess.run(clf.model.b))
    print(clf.predict(-2))
    print(clf.predict(4))

    del clf
    print('\ndel model ...')
    clf = BiClassifier()
    clf.load('/tmp/biclassifier_model', global_step=61)
    print('loaded weight: %f' % clf.sess.run(clf.model.b))
    print(clf.predict(-2))
    print(clf.predict(4))

