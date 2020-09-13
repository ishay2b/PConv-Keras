import tensorflow as tf
import tensorflow
from google.protobuf import text_format
from tensorflow.compat.v1 import GraphDef
import cv2

import numpy as np
from numpy import array as _A 
import sys
import os
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import requests
import json
from tensorflow.keras import backend as K
from timeit import default_timer

from libs.pconv_model import PConvUnet

# Change to root path

from pathlib import Path as _P

RESIZED = 512

ROOT = _P(os.path.abspath(__file__)).parents[1]
CONV_ROOT = _P(os.path.abspath(__file__)).parents[0]
MODAL_PATH = str(ROOT / "PConv-Keras/weights/pconv_imagenet.26-1.07.h5")
EXPORT_PATH = ROOT/ "exported_models" / "pconv_imagenet" / "2"

#model_dir = ROOT / "exported_models/frozen/model"


# Let's read our pbtxt file into a Graph protobuf
if 0:
    model_path = '/home/ishay/projects/EntitiMed/exported_models/estimator2/saved_model.pbtxt'
    f = open(model_path, "rb")
    graph_protobuf = text_format.Parse(f.read(), GraphDef())

    # Import the graph protobuf into our new graph.
    graph_clone = tf.Graph()
    with graph_clone.as_default():
        tf.import_graph_def(graph_def=graph_protobuf, name="")
start_time = default_timer()
pb_path = EXPORT_PATH / 'saved_model.pb'
pb_path_txt = EXPORT_PATH / 'saved_model.pbtxt'
net = cv2.dnn.readNetFromTensorflow(str(pb_path), str(pb_path_txt))
net = cv2.dnn.readNetFromTensorflow('', str(pb_path_txt))
print("Loading took %f seconds" %(default_timer() - start_time))

# Input image
img = cv2.imread(str(ROOT / "InPaintingGroundTruth/patient9/Generated/Brush_25/Original.png"))
rows, cols, channels = img.shape

# Use the given image as input, which needs to be blob(s).
blob1 = cv2.dnn.blobFromImage(img, size=(512, 512), swapRB=True, crop=False)
blob2 = cv2.dnn.blobFromImage(img, size=(512, 512), swapRB=True, crop=False)

net.setInput(blob1, "inputs_img")
net.setInput(blob2, "inputs_mask")

