import tensorflow as tf

MODEL_DIR = './model'
TAG = tf.saved_model.tag_constants.SERVING
SIGNATURE_KEY = (
    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
IN_TENSOR_KEY = tf.saved_model.signature_constants.PREDICT_INPUTS
OUT_TENSOR_KEY = tf.saved_model.signature_constants.PREDICT_OUTPUTS
EMBEDDING_DIMENSION = 15
NUM_ANNOY_TREES = 10