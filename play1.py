import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input

inception_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
inception_model.save('inception.h5')

from tensorflow.keras import backend as K
import tensorflow as tf
# The export path contains the name and the version of the model
tensorflow.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model = tf.keras.models.load_model('./inception.h5')
export_path = '../my_image_classifier/1'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})


'''
tensorflow_model_server --model_base_path=/home/ishay/projects/my_image_classifier --rest_api_port=9111 --model_name=ImageClassifier
'''
from tensorflow.keras.preprocessing import image
import requests

at:localhost:9111
r = requests.post('http://localhost:9111/v1/models/ImageClassifier:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

