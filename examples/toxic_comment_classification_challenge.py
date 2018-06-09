#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd

from sklearn_plus.preprocessing.bag_of_words import TextToBagVec
from sklearn_plus.preprocessing.label_onehot_encoder import LabelOneHotEncoder
from sklearn_plus.natural_language_processing.text_classification import Estimator as TextClassifier
from sklearn_plus.natural_language_processing.text_classification.model_fn import text_cnn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

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
                batch_size = 20,
                num_epochs = 1,
                max_document_length = 20
            )
 
    clf = TextClassifier(model_dir='./model/', model_fn=text_cnn, params=params)

    clf.fit(X, y)

    print clf.predict_top_n(X[:100])
