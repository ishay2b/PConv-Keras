import os
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
INPUT_SIZE = 513

model_path = '/home/ishay/projects/EntitiMed/exported_models/pconv_imagenet/1/saved_model.pb'

f = tf.io.gfile.GFile(model_path, 'rb')
txt = f.read(-1)
graph_def = tf.compat.v1.GraphDef()
graph_def.ParseFromString(txt)
    g_in = tf.import_graph_def(graph_def, name="")
sess = tf.Session(graph=g_in)

