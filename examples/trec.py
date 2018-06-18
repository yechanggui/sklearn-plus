#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np

from sklearn_plus.natural_language_processing.text_classification import Estimator as TextClassifier
from sklearn_plus.natural_language_processing.text_classification.model_fn import text_cnn

import tensorflow as tf

# dataset: http://cogcomp.org/Data/QA/QC/
if __name__ == '__main__':

    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)        #Setup format, level = DEBUG, INFO, WARNING, ERROR, CRITICAL

    tf.logging.set_verbosity(tf.logging.INFO)

    # load raw data
    train_raw_data_file = 'data/trec/train_5500.label.txt'

    X = []
    y = []
    with open('data/trec/train_5500.label.txt') as f:
        for l in f:
            v = l.strip().split(':')
            X.append(':'.join(v[1:]))
            y.append(v[0])

    params = tf.contrib.training.HParams(
            filter_sizes = [1,2,3,4,5,6,7],
            embed_dim = 100,
            num_filters = 256,
            dropout_keep_prob = 0.5,
            l2_lambda = 0.0001,
            decay_steps = 6000,
            decay_rate = 0.65,
            batch_size = 256,
            num_epochs = 10,
            learning_rate = 0.01,
            max_document_length = 70
        )
     
    clf = TextClassifier(model_dir='./model/', model_fn=text_cnn, params=params)

    clf.fit(X, y)

    cor_cnt = 0.0
    all_cnt = 0.0
    with open('data/trec/TREC_10.label.txt') as f:
        for l in f:
            all_cnt += 1
            v = l.strip().split(':')
            if clf.predict([':'.join(v[1:])])[0] == v[0]:
                cor_cnt += 1

    # 0.978
    logging.info(cor_cnt / all_cnt)
