#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import datetime
import os

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn_plus.utils.data_helpers import batch_iter
from sklearn_plus.preprocessing.label_onehot_encoder import LabelOneHotEncoder

from .models.cnn_lstm import CNN_LSTM  # OPTION 0
from .models.lstm_cnn import LSTM_CNN  # OPTION 1
from .models.cnn import CNN  # OPTION 2 (Model by: Danny Britz)
from .models.lstm import LSTM  # OPTION 3
from ...base import ModelMixin  #

import tensorflow as tf
from tensorflow.contrib import learn

import numpy as np


# Source: https://github.com/pmsosa/CS291K
class TextClassifierCP(BaseEstimator, ClassifierMixin, ModelMixin):

    def __init__(self, checkpoint_dir=None, summary_dir=None):
        super(TextClassifierCP, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.summary_dir = summary_dir

    def fit(self, X, y=None):

        MODEL_TO_RUN = 0

        # Data loading params
        dev_size = .10

        # Model Hyperparameters
        embedding_dim = 32  # 128
        max_seq_legth = 70
        filter_sizes = [3, 4, 5]  # 3
        num_filters = 32
        dropout_prob = 0.5  # 0.5
        l2_reg_lambda = 0.0
        use_glove = False  # Do we use glove

        # Training parameters
        batch_size = 128
        num_epochs = 10  # 200

        x_text = X

        self.onehotencoder = LabelOneHotEncoder()
        y = self.onehotencoder.fit_transform(y)

        # Build vocabulary
        max_document_length = max([len(x.split(' ')) for x in x_text])
        if (not use_glove):
            print "Not using GloVe"
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            x = np.array(list(self.vocab_processor.fit_transform(x_text)))
        else:
            print "Using GloVe"
            embedding_dim = 50
            filename = '../glove.6B.50d.txt'

            def loadGloVe(filename):
                vocab = []
                embd = []
                file = open(filename, 'r')
                for line in file.readlines():
                    row = line.strip().split(' ')
                    vocab.append(row[0])
                    embd.append(row[1:])
                print('Loaded GloVe!')
                file.close()
                return vocab, embd

            vocab, embd = loadGloVe(filename)
            vocab_size = len(vocab)
            embedding_dim = len(embd[0])
            embedding = np.asarray(embd)

            W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                            trainable=False, name="W")
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            embedding_init = W.assign(embedding_placeholder)

            self.sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

            # init vocab processor
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            # fit the vocab from glove
            pretrain = self.vocab_processor.fit(vocab)
            # transform inputs
            x = np.array(list(self.vocab_processor.transform(x_text)))

            # init vocab processor
            self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
            # fit the vocab from glove
            pretrain = self.vocab_processor.fit(vocab)
            # transform inputs
            x = np.array(list(self.vocab_processor.transform(x_text)))

        # Randomly shuffle data
        np.random.seed(42)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(dev_size * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        # Training
        # ==================================================

        # embed()
        if (MODEL_TO_RUN == 0):
            print x_train.shape
            print y_train.shape
            self.model = CNN_LSTM(x_train.shape[1], y_train.shape[1], len(self.vocab_processor.vocabulary_),
                                  embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 1):
            self.model = LSTM_CNN(x_train.shape[1], y_train.shape[1], len(self.vocab_processor.vocabulary_),
                                  embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 2):
            self.model = CNN(x_train.shape[1], y_train.shape[1], len(self.vocab_processor.vocabulary_),
                             embedding_dim, filter_sizes, num_filters, l2_reg_lambda)
        elif (MODEL_TO_RUN == 3):
            self.model = LSTM(x_train.shape[1], y_train.shape[1], len(self.vocab_processor.vocabulary_),
                              embedding_dim)
        else:
            print "PLEASE CHOOSE A VALID MODEL!\n0 = CNN_LSTM\n1 = LSTM_CNN\n2 = CNN\n3 = LSTM\n"
            exit();

            # Define Training procedure
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        # Write vocabulary
        self.vocab_processor.save(os.path.join(self.checkpoint_dir, "vocab"))

        # init variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        summaries_dict = {
            "loss":self.model.loss,
            "accuracy":self.model.accuracy
        }

        # TRAINING STEP
        def train_step(x_batch, y_batch, save=False,summaries_dict=None):
            feed_dict = {
                self.model.input_x: x_batch,
                self.model.input_y: y_batch,
                self.model.dropout_keep_prob: dropout_prob
            }
            _, step, loss, accuracy = self.sess.run(
                [train_op, self.global_step, self.model.loss, self.model.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            if step % 20 == 0:
                # if summary is needed
                if self.summary_dir:
                    self.summaries(self.summary_dir, feed_dict, summaries_dict)

                # if checkpoint is needed
                if self.checkpoint_dir:
                    self.save(self.checkpoint_dir)

        # CREATE THE BATCHES GENERATOR
        batches = batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

        # TRAIN FOR EACH BATCH
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch,summaries_dict=summaries_dict)



    def predict(self, X):
        feed_dict = {
            self.model.input_x: np.array(list(self.vocab_processor.transform(X))),
            self.model.dropout_keep_prob: 0.5
        }
        prediction_result = self.sess.run(
            self.model.prediction,
            feed_dict)
        return prediction_result


    def predict_proba(self, X):
        feed_dict = {
            self.model.input_x: np.array(list(self.vocab_processor.transform(X))),
            self.model.dropout_keep_prob: 0.5
        }
        logits = self.sess.run(
            self.model.logits,
            feed_dict)
        return logits.tolist()
