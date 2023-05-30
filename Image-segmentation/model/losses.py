import tensorflow as tf


def mse_loss_function(labels, preds):
    preds = tf.expand_dims(tf.reduce_sum(preds, -1), -1)
    labels = tf.where(labels > 0.0, 1., .0)
    l2 = tf.subtract(labels, preds)
    l2 = tf.square(l2)
    l2 = tf.reduce_sum(l2) / tf.cast(tf.size(labels), tf.float32)
    return l2


def _piece_wise_cross(label, pred):
    return -tf.reduce_sum(label * tf.math.log(pred + 1e-5) +
                          (1 - label) * tf.math.log(1 - pred + 1e-5)) / tf.cast(tf.size(label), tf.float32)


def piece_wise_cross(label, pred):
    y1 = tf.where(label >= .5, 1., 0.)
    l1 = _piece_wise_cross(y1, pred)
    return l1


def regulator_l1(volumen, alpha=1.5e-7):  # 4e-6, 2.5e-6
    pixel_dif1 = volumen[:, 1:, :, :] - volumen[:, :-1, :, :]
    pixel_dif2 = volumen[:, :, 1:, :] - volumen[:, :, :-1, :]
    total_var = tf.reduce_sum(
        tf.reduce_sum(tf.abs(pixel_dif1)) +
        tf.reduce_sum(tf.abs(pixel_dif2))
    )
    return total_var * alpha


def _dice_loss(x, y):
    num = 2 * tf.reduce_sum(x * y)
    den = tf.reduce_sum(y ** 2) + tf.reduce_sum(x ** 2) + 1e-3
    return 1 - num / den


def dice_loss(label, pred):
    y1 = tf.where(label >= .5, 1., 0.)
    l1 = _dice_loss(y1, pred)
    return tf.where(tf.math.is_nan(l1), 0., l1)
