#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn_plus.preprocessing.text.zh import Normalizer
from sklearn_plus.preprocessing.text.zh import JiebaTokenizer
from sklearn_plus.preprocessing.text.en import PuncTokenizer

import sys
import logging
from optparse import OptionParser
import pandas as pd

if __name__ == '__main__':

    FORMAT = '[%(asctime)-15s] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = OptionParser()
    parser.add_option("--training_data", dest = "training_data", metavar = "FILE", help = "training data")
    parser.add_option("--validation_data", dest = "validation_data", metavar = "FILE", help = "validation data")
    parser.add_option("--model_dir", dest = "model_dir", metavar = "FILE", help = "model dir")
    parser.add_option("--mode", dest="mode", help="interaction mode: training, test")

    if len(sys.argv) == 1:
        parser.print_help()
        exit()

    (options, args) = parser.parse_args()

    df = pd.read_csv(options.training_data, dtype=object)

    clfs = []
    for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        logging.info("processing " + label + "...")
        clf = Pipeline([('token', PuncTokenizer()),
                        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1, smooth_idf=1, sublinear_tf=1)),
                        ('clf', LogisticRegression(C=4, dual=True)),
                       ])
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
    df_out = pd.DataFrame(df_out_list, columns=('id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'))
    df_out.to_csv(options.validation_data + '.out', index=False)
