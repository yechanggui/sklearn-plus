#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn_plus.preprocessing.text.en import PuncTokenizer
from sklearn_plus.neural_network.text_classification.text_classifier_cp import TextClassifier

import sys
import logging
from optparse import OptionParser
import pandas as pd

if __name__ == '__main__':

    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = OptionParser()
    parser.add_option("--training_data", dest="training_data", metavar="FILE", help="training data")
    parser.add_option("--validation_data", dest="validation_data", metavar="FILE", help="validation data")
    parser.add_option("--model_dir", dest="model_dir", metavar="FILE", help="model dir")
    parser.add_option("--mode", dest="mode", help="interaction mode: training, test")

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    (options, args) = parser.parse_args()

    df = pd.read_csv(options.training_data, dtype=object)

    clfs = []
    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        logging.info("processing " + label + "...")
        clf = Pipeline([('clf', TextClassifier(checkpoint_dir='/Users/liuxiaoan/Downloads/sklearn_plus_test_cp',
                                               summary_dir="/Users/liuxiaoan/Downloads/sklearn_plus_test_cp"))])
        clf.fit(df['comment_text'], df[label])
        clfs.append(clf)

    df = pd.read_csv(options.validation_data, dtype=object)

    df_out_list = []
    for i, row in df.iterrows():
        new_item = [row['id']]
        for clf in clfs:
            predicted = clf.predict_proba([row['comment_text']])
            new_item.append(predicted[0][1])
        df_out_list.append(new_item)
    df_out = pd.DataFrame(df_out_list,
                          columns=('id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'))
    df_out.to_csv(options.validation_data + '.out', index=False)
