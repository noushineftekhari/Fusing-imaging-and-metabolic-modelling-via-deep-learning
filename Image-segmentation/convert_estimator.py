import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from model.estimator import Model

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.random.set_seed(1024)

version = '4'
window = [512, 512]


version = '-{}'.format(version)

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 8
epochs = 2 ** 17
learning_rate = 1e-3     # 1e-3
shuffles = 12

model = Model(learning_rate, window, version, batch)
input_column = tf.feature_column.numeric_column("x", shape=(*window, 1))


def serving_input_fn():
    features = {'x': tf.compat.v1.placeholder(shape=[1, *window, 1], dtype=tf.float32)}
    return tf.estimator.export.ServingInputReceiver(features, features)


estimator_path = model.model.export_saved_model('saved_model', serving_input_fn)

# imported = tf.saved_model.load('saved_model/model')
#
#
# def predict(x):
#   return imported.signatures["serving_default"](x)
#
# print(predict(tf.zeros((1, *window, 3)))['output'].shape)

# converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/model')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
#
# quantized_tflite_model = converter.convert()












