from __future__ import absolute_import

from .text_classification.models import *


def deserialize(class_name):
    return globals()[class_name]
