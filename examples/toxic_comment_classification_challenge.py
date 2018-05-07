#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn_plus.preprocessing.bag_of_words import TextToBagVec
from sklearn_plus.preprocessing.label_onehot_encoder import LabelOneHotEncoder
from sklearn_plus.neural_network.text_classification.text_classifier import TextClassifier

if __name__ == '__main__':
    # load raw data
    train_raw_data_file = './data/train.csv'
    df = pd.read_csv(train_raw_data_file, dtype=object)
    X = df['comment_text']
    y = df['toxic']
    print('raw data is ready...')

    # pre-processing
    textencode = TextToBagVec()
    vocab_processor = textencode.fit(X)
    train_data = textencode.transform(X)
    print('train data is ready...')

    onehotencoder = LabelOneHotEncoder()
    train_label = onehotencoder.fit_transform(y)
    print('label is ready...')

    # train model
    clf = TextClassifier(len(vocab_processor.vocabulary_), checkpoint_dir='./classifier_model',
                         summary_dir='./classifier_model')
    clf.fit(train_data, train_label)
    print('done.')

    # save model
    clf.save('./classifier_model')

    # load model
    del clf
    clf = TextClassifier(len(vocab_processor.vocabulary_), checkpoint_dir='./classifier_model',
                         summary_dir='./classifier_model')
    clf.load('./classifier_model')

    # prediction
    print(clf.predict(train_data[:10]))
    print(clf.predict_proba(train_data[:10]))
    print(train_label[:10])
