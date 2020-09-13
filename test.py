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


from libs.pconv_model import PConvUnet

# Change to root path

from pathlib import Path as _P

RESIZED = 512

ROOT = _P(os.path.abspath(__file__)).parents[1]
CONV_ROOT = _P(os.path.abspath(__file__)).parents[0]
MODAL_PATH = str(ROOT / "PConv-Keras/weights/pconv_imagenet.26-1.07.h5")
EXPORT_PATH = str(ROOT/ "exported_models" / "pconv_imagenet" / "2")

def prepare_image(image_tensor, mask_tensor):
    #image_contents = tf.read_file(image_str_tensor)
    #image = tf.image.decode_jpeg(image_contents, channels=3)
    #image = tf.image.resize_images(image, [224, 224])
    image = tf.cast(image_tensor, tf.float32)
    return image / 255.0
    #return preprocess_input(image)

def serving_input_receiver_fn():
    inputs_img = tf.placeholder(tf.float32, shape=[None, RESIZED, RESIZED, 3], name='inputs_img')
    inputs_mask = tf.placeholder(tf.float32, shape=[None, RESIZED, RESIZED, 3], name='inputs_mask')
    #images_tensor = tf.map_fn(prepare_image, [inputs_img, inputs_mask], back_prop=False, dtype=tf.float32)
    #images_tensor = tf.image.convert_image_dtype(images_tensor, dtype=tf.float32)

    return tf.estimator.export.ServingInputReceiver({"inputs": [{"inputs_img":inputs_img},{"inputs_mask":inputs_mask}]})



class InPantingEngine(object):
    def __init__(self, model_path=MODAL_PATH):
        self.model_path = model_path
        tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
        self.model = PConvUnet(vgg_weights=None, inference_only=True)
        tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
        self.model.load(self.model_path)
    
    def preprocess(self, image, mask):
        im = _A(image, dtype=np.float32) / 255.0
        msk = _A(mask, dtype=np.float32) / 255.0
        msk = 1.0 - msk
        im[msk==0] = 1
        return im, msk

    def __call__(self, image, mask):
        im , msk = self.preprocess(image, mask)
        pred_imgs = self.model.predict([[im], [msk]])[0]
        pred_imgs = (pred_imgs * 255).astype(np.uint8)
        return pred_imgs

    def dump_to_pb(self):
        import shutil
        # The export path contains the name and the version of the model

        # Fetch the Keras session and save the model
        # The signature definition is defined by the input and output tensors
        # And stored with the default serving key
        try:
            shutil.rmtree(EXPORT_PATH)
            print("Deleted prevoius export path", EXPORT_PATH)
        except:
            pass
        with tf.keras.backend.get_session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.saved_model.simple_save(
                sess,
                EXPORT_PATH,
                inputs={'inputs_img': self.model.model.inputs[0], "inputs_mask": self.model.model.inputs[1]},
                outputs={t.name: t for t in self.model.model.outputs})

    def dump_to_estimator(self):
        self.model_dir = str(ROOT/'exported_models' / 'estimator2')
        tf.io.write_graph(self.model.model.output.graph,self.model_dir, 'saved_model.pbtxt',as_text=True)
        tf.io.write_graph(self.model.model.output.graph,self.model_dir, 'saved_model.pb',as_text=False)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        session.run(tf.global_variables_initializer())
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


if __name__ == '__main__':

    self = InPantingEngine()
    #self.dump_to_pb()
    #self.dump_to_estimator()

    frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in self.model.model.outputs])
    tf.train.write_graph(frozen_graph, EXPORT_PATH, "saved_model.pb", as_text=False)
    tf.train.write_graph(frozen_graph, EXPORT_PATH, "saved_model.pbtxt", as_text=True)

    raise BaseException("Done")

    image_path = '/home/ishay/projects/EntitiMed/InPaintingGroundTruth/patient9/Generated/Brush_25/Original.png'
    mask_path = '/home/ishay/projects/EntitiMed/InPaintingGroundTruth/patient9/Generated/Brush_25/Mask.png'

    image = cv2.imread(image_path)
    cv2.imshow('Original', image)
    
    mask = cv2.imread(mask_path)
    cv2.imshow('mask', mask)
    both = image+mask
    cv2.imshow('Original+mask', both)

    pred_imgs = self(image, mask)
    #reconstructed_image = chunker.dimension_postprocess(pred_imgs, image)
    cv2.imshow('reconstructed_image', pred_imgs)
    cv2.waitKey(10)






    #export saved_model_path='/home/ishay/projects/EntitiMed/exported_models/pconv_imagenet'

    #tensorflow_model_server --model_base_path=$saved_model_path --rest_api_port=9222 --model_name=pconv_imagenet 

    image_path = '/home/ishay/projects/EntitiMed/InPaintingGroundTruth/patient9/Generated/Brush_25/Original.png'
    mask_path = '/home/ishay/projects/EntitiMed/InPaintingGroundTruth/patient9/Generated/Brush_25/Mask.png'

    # Preprocessing our input image
    img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.
    msk = image.img_to_array(image.load_img(mask_path, target_size=(224, 224))) / 255.

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')
    msk = msk.astype('float16')


    # sending post request to TensorFlow Serving server
    r = requests.post('http://localhost:9222/v1/models/pconv_imagenet:predict', json={"instances": [{'inputs_img_1:0': img.tolist(), 'inputs_mask_1:0':msk.tolist()}]})
    pred = json.loads(r.content.decode('utf-8'))

