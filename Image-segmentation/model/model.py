import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import concatenate, LeakyReLU

# size = 32
size = 24


def multy_layer_2(inputs, number_filter):
    layer_a = Conv2D(number_filter, kernel_size=3, padding='same', strides=2)(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv2D(number_filter // 2, kernel_size=1, padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    return BatchNormalization()(layer_a)


def multy_layer(inputs, number_filter):
    layer_a = Conv2D(number_filter, kernel_size=3, padding='same')(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv2D(number_filter // 2, kernel_size=1, padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    return BatchNormalization()(layer_a)


def deconv_layer(inputs, number_filter):
    layer_a = Conv2DTranspose(number_filter, kernel_size=3, strides=2, padding='same')(inputs)
    layer_a = LeakyReLU()(layer_a)
    layer_a = Conv2D(number_filter // 2, kernel_size=1, padding='same')(layer_a)
    layer_a = LeakyReLU()(layer_a)
    return BatchNormalization()(layer_a)


class Network:
    def main(self, inputs):

        inputs = tf.reshape(inputs, [-1, 512, 512, 1])

        l1 = multy_layer(inputs, size // 2)
        l1 = multy_layer(l1, size // 2)
        l2 = multy_layer_2(l1, 2 * size // 2)
        l3 = multy_layer_2(l2, 4 * size // 2)
        l4 = multy_layer_2(l3, 8 * size // 2)

        l5 = multy_layer_2(l4, 16 * size)

        l6 = deconv_layer(l5, 8 * size)
        l6 = concatenate([l4, l6], axis=-1)
        l6 = multy_layer(l6, 8 * size)

        l7 = deconv_layer(l6, 4 * size)
        l7 = concatenate([l3, l7], axis=-1)
        l7 = multy_layer(l7, 4 * size)

        l8 = deconv_layer(l7, 2 * size)
        l8 = concatenate([l2, l8], axis=-1)
        l8 = multy_layer(l8, 2 * size)

        l9 = deconv_layer(l8, size)
        l9 = concatenate([l1, l9], axis=-1)
        l9 = multy_layer(l9, size)
        l9 = Conv2D(size // 2, kernel_size=3, padding='same')(l9)
        l9 = LeakyReLU()(l9)

        output = Conv2D(1, 1, padding='same')(l9)
        return output
