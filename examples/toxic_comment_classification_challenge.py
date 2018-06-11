#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd

from sklearn_plus.natural_language_processing.text_classification import Estimator as TextClassifier
from sklearn_plus.natural_language_processing.text_classification.model_fn import text_cnn

import tensorflow as tf

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.INFO)

    # load raw data
    train_raw_data_file = './data/train.csv.head'
    df = pd.read_csv(train_raw_data_file, dtype=object)
    X = df['comment_text']
    y = df['toxic']
    print('raw data is ready...')

    params = tf.contrib.training.HParams(
                filter_sizes = [1,2,3,4,5,6,7],
                embed_dim = 100,
                num_filters = 256,
                dropout_keep_prob = 0.5,
                l2_lambda = 0.0001,
                decay_steps = 6000,
                decay_rate = 0.65,
                batch_size = 20,
                num_epochs = 1,
                learning_rate = 0.01,
                max_document_length = 20
            )
 
    clf = TextClassifier(model_dir='./model/', model_fn=text_cnn, params=params)

    clf.fit(X, y)

    print clf.predict_top_n(X[:100])
