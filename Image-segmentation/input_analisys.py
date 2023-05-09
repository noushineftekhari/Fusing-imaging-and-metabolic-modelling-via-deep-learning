import os
import tensorflow as tf
from data.tfrecords import read_and_decode
import numpy as np
from data.tf_augmentation import tf_random_bright, tf_norm, tf_translation, tf_rotation, tf_random_zoom_in_out, tf_flip


physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.random.set_seed(1024)

version = '1'
window = [512, 512]
classes = 1
path = r'D:\Unet\tfrecords\\'

train_path = list(os.walk(path + 'train\\'))[0][-1]
train_path = [path + 'train\\' + x for x in train_path]


# def train_inputs(batch_size=1, num_shuffles=1, para=24):
#     dataset = read_and_decode(train_path)
#     dataset = dataset.batch(batch_size=batch_size)
#     dataset = dataset.prefetch(buffer_size=24)
#     return dataset

def train_inputs(batch_size=1, num_shuffles=1, para=24):
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
    dataset = dataset.map(lambda x, y: tf_rotation(x, y), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: tf_translation(x, y), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: (tf_norm(x), y), num_parallel_calls=para)
    dataset = dataset.map(lambda x, y: tf_random_bright(x, y, [0.95, 1.05]), num_parallel_calls=para)
    dataset = dataset.shuffle(num_shuffles)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=24)
    return dataset


count = 0
data = train_inputs()
try:
    for x, y in data.take(33):
        count += 1
        print(x.shape, y.shape)
        # if np.max(x) != np.max(y):
            # print(np.max(x), np.max(y))
        # print(np.max(y))
            # print(train_path[count])

except:
    print(train_path[count])
