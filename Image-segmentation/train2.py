import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import tensorflow as tf
from data.tfrecords import read_and_decode
from data.tf_augmentation import tf_random_bright, tf_norm, tf_translation, tf_rotation, tf_random_zoom_in_out, tf_flip
from model.estimator import Model

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.random.set_seed(1024)


# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# os.environ['TF_ENABLE_MIXED_PRECISION'] = '1'
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# tf.config.optimizer.set_jit(True)

version = '4'
window = [512, 512]
classes = 1
path = r'D:\Unet\tfrecords\3\\'
# programdir = os.path.abspath(os.path.dirname(sys.argv[0]))

version = '-{}'.format(version)
train_path = list(os.walk(path + 'train\\'))[0][-1]
test_path = list(os.walk(path + 'eval\\'))[0][-1]

train_path = [path + 'train\\' + x for x in train_path]
test_path = [path + 'eval\\' + x for x in test_path]

# set the path's were you want to storage the data(tensorboard and checkpoints)
batch = 8
epochs = 2 ** 17
# learning_rate = 1.5e-5     # 1e-3
learning_rate = 1.2e-5     # 5e-6
shuffles = 12


def train_inputs(batch_size=1, num_shuffles=shuffles, para=24):
    """

    Parameters
    ----------
    batch_size
    num_shuffles
    para

    Returns
    -------

    """
    dataset = read_and_decode(train_path)
    dataset = dataset.repeat(4)
    dataset = dataset.map(lambda x, y: tf_flip(x, y, 1), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: tf_flip(x, y, 0), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: tf_random_zoom_in_out(x, y, window, [.8, 1.2], [1, 1]), num_parallel_calls=para)
    # dataset = dataset.map(lambda x, y: tf_rotation(x, y), num_parallel_calls=para)
    # dataset = dataset.map(lambda x, y: tf_translation(x, y), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: (tf_norm(x), y), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: tf_random_bright(x, y, [0.9, 1.1]), num_parallel_calls=para)
    dataset = dataset.shuffle(num_shuffles)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=24)
    return dataset


def eval_inputs():
    """

    Returns
    -------

    """
    dataset = read_and_decode(test_path)
    dataset = dataset.map(lambda x, y: (tf_norm(x), y), num_parallel_calls=24)
    dataset = dataset.shuffle(shuffles).batch(batch)
    return dataset


for i in range(10000):
    # print('training iter {}'.format(i))
    estimator = Model(learning_rate, window, version, batch)
    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=epochs * 16)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=100000)
    # train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_inputs(batch), max_steps=200)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_inputs())
    estimator.train(train_spec, eval_spec)
