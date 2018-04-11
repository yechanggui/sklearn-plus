#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn_plus.neural_network.text_classification import DemoClassifier

if __name__ =="__main__":
    # clf = TextClassifier(checkpoint_dir='/Users/liuxiaoan/Downloads/sklearn_plus_test_cp',
    #                summary_dir="/Users/liuxiaoan/Downloads/sklearn_plus_test_cp")
    # clf.load('/Users/liuxiaoan/Downloads/sklearn_plus_test_cp')
    # df = pd.read_csv('/Users/liuxiaoan/Downloads/test.csv', dtype=object)
    #
    # print(clf.predict(df['comment_text'][:10]))
    x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
    y_vals = np.concatenate((np.repeat(-1., 50), np.repeat(1., 50)))
    print 'data is ready...'

    clf = DemoClassifier(summary_dir='/Users/liuxiaoan/Downloads/sklearn_plus_test',
                         checkpoint_dir='/Users/liuxiaoan/Downloads/sklearn_plus_test')
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

    del clf
    print 'del model ...'
    clf = DemoClassifier()
    print clf.load('/Users/liuxiaoan/Downloads/sklearn_plus_test', global_step=1)
    print 'loaded weight: %f' % clf.sess.run(clf.model.A)
    print clf.predict(-2)
    print clf.predict(4)
