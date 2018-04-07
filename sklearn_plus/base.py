#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base classes for all neural networks."""
import os
import tensorflow as tf


###############################################################################
class ModelMixin(object):
    """Mixin class for all neural networks in sklearn_plus."""

    def __init_saver(self,out_dir):
        checkpoint_prefix = os.path.join(out_dir, "model")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.checkpoint_prefix = checkpoint_prefix
        self.saver = tf.train.Saver()

    def save(self,out_dir):
        """save model weights to out_dir.

        This method save a tensorflow model's weights to specified directory.

        Parameters
        ----------
        out_dir : string
             a string like './models' or '/home/username/models'

        Returns
        -------
        path : string
            model saved path.

        """
        if not hasattr(self,'saver'):
            self.__init_saver(out_dir)
        return self.saver.save(self.sess, self.checkpoint_prefix)

    def load(self, model_dir):
        """restore model weights from model directory.

        This method load model weights from model directory and restore
        the model.

        Parameters
        ----------
        model_dir : string
            a string like './models' or '/home/username/models'

        """
        if not hasattr(self, 'saver'):
            self.__init_saver(model_dir)
        return self.saver.restore(self.sess, self.checkpoint_prefix)
