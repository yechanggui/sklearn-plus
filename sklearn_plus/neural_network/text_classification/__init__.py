from .models import *

def deserialize(class_name):
    return globals()[class_name]
