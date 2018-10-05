import tensorflow as tf


def awgn(tx, N0, **kwargs):
    name = kwargs.get('name', 'AWGN')
    with tf.name_scope(name):
        w = tf.random_normal(tf.shape(tx), 0., 10. ** (N0 / 10.),
                             name='WGN')
        rx = tf.add(tx, w, name='AWGN_out')
    return rx


def normalize_power(x, **kwargs):
    name = kwargs.get('name', 'power_normalizer')
    with tf.name_scope(name):
        tx = tf.subtract(x, tf.reshape(tf.reduce_mean(x, 1), (-1, 1, 2)))
        tx = tf.divide(tx, tf.sqrt(
            tf.reshape(tf.reduce_mean(tf.pow(tx, 2.), 1), (-1, 1, 2))),
                       name='normalized_signal')
    return tx


def accuracy(x, x_hat, **kwargs):
    name = kwargs.get('name', 'accuracy')
    return tf.reduce_mean(tf.cast(tf.equal(x_hat, x), tf.float32), name=name)


def power(tx, **kwargs):
    name = kwargs.get('name', 'power')
    return tf.reduce_mean(tf.pow(tx, 2.), name=name)


# loss functions

def sigmoid_cross_entropy_loss(y, y_hat, **kwargs):
    name = kwargs.get('name', 'sigmoid_cross_entropy')
    return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y, logits=y_hat
                ), name=name
            )
