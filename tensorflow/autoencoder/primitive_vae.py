
import os
import numpy as np
import scipy.signal as sig
import tensorflow as tf
from tensorflow.python.layers.layers import conv1d, dense
from autoencoder import Autoencoder
from neural_network import NeuralNetwork
import tensorflow_utilities as tf_util

from tensorflow.examples.tutorials.mnist import input_data

from genfigures.plot_functions import get_index_list

# The basic idea here is that the autoencoder is learning a function approxation
# rather than the identity function.
# This allows it to learn an inverse transform that can be used as a channel for transmitting

class VAE(Autoencoder):
    def __init__(self, input_dim, latent_size, output_dim, inner_width=50, lr=1e-4, **kwargs):
        if 'beta' in kwargs:
            beta = kwargs['beta']
        else:
            beta = 1.*latent_size/input_dim

            # input
        x = tf.placeholder(tf.float32, (None, input_dim))
        self.x = x
        self.target = tf.placeholder(tf.float32, (None, output_dim))
        #self.y = y
        # encoder
        with tf.name_scope('encoder'):
            enc1 = dense(x, inner_width, activation=tf.nn.relu, name='enc1')
            enc2 = dense(enc1, inner_width, activation=tf.nn.relu, name='enc2')
            enc3 = dense(enc2, inner_width, activation=tf.nn.relu, name='enc3')
            #enc4 = dense(enc3, inner_width, activation=tf.nn.relu, name='enc4')

            # variational layers
            mn = dense(enc3, units=latent_size)
            sd       = 0.5 * tf.layers.dense(enc3, units=latent_size)
            epsilon = tf.random_normal(tf.stack([tf.shape(enc3)[0], latent_size]))

            samples  = mn + tf.multiply(epsilon, tf.exp(sd))
            self.tx = mn
            self.tx_std = sd
            #self._latent_in = samples
            self.rx = samples
            #self.rx = tf.concat((samples, self.start_state), axis=1) # starting state feeds into the latent space

        # decoder
        with tf.name_scope('decoder'):
            # NOTE: following line is tightly coupled to the architecture
            # dec1 is of size (None, 236, 4)
            dec1 = dense(self.rx, inner_width, activation=tf.nn.relu, name='dec1')
            dec2 = dense(dec1, inner_width, activation=tf.nn.relu, name='dec2')
            dec3 = dense(dec2, inner_width, activation=tf.nn.relu, name='dec3')
            #dec4 = dense(dec3, inner_width, activation=tf.nn.relu, name='dec4')

            x_out = dense(dec3, output_dim, activation=tf.nn.relu, name='rx_out')
            self.x_out = x_out

        # quality metrics
        with tf.name_scope('metrics'):
            # change to MSE for accuracy, tx_power has no bearing
            #self.accuracy = tf.reduce_sum(tf.squared_difference(x_out, self.target))#tf_util.accuracy(x, self.rx_bits, name='accuracy')
            self.accuracy = tf.losses.absolute_difference(x_out, self.target)
            self.latent_loss = -0.5 * tf.reduce_mean(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd))

            #self.tx_power = tf_util.power(tx, name='tx_power')
        # loss
        with tf.name_scope('loss'):

            #art_loss = tf.reduce_sum(tf.squared_difference(x_out, self.target), 1)
            art_loss = tf.losses.absolute_difference(x_out, self.target)

            latent_loss = -0.5 * tf.reduce_mean(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd))
            latent_loss = -0.5 * tf.reduce_mean(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            #latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)

            #loss = tf.reduce_sum(art_loss + beta*latent_loss)
            loss = tf.reduce_mean(art_loss + beta*latent_loss)
            #loss = tf.reduce_mean(art_loss + latent_size/input_dim*latent_loss)
            #loss = tf.reduce_mean(art_loss + input_dim/latent_size*latent_loss)
            #loss = tf.reduce_mean(art_loss + latent_loss)

            self.loss = loss
        # optimizer
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        # initialize the base class
        #metrics = [self.loss, self.accuracy, self.tx_power]
        metrics = [self.loss, self.accuracy, self.latent_loss]
        encoder = NeuralNetwork(x, self.tx)
        self.encoder_std = NeuralNetwork(x, self.tx_std) # get the deviation on the encoder output
        decoder = NeuralNetwork(self.rx, self.x_out)
        Autoencoder.__init__(self, encoder, decoder, self.train_step, metrics=metrics, **kwargs)

    def encode_std(self, x):
        return self.sess.run(self.encoder_std.output, {self.encoder_std.input: x})

    # note: self.target must get defined in the inheriting class
    #def get_feed_dict(self, dset, n=None, batch_size=None):
    #    if n is None or batch_size is None:
    #        x, target, state = dset.data()
    #    else:
    #        x, target, state = dset.get_batch(n, batch_size)
    #    return {self.x: x, self.target: target, self.start_state: state}

    #def encode(self, x):
    #    return self.sess.run(self.encoder.output, {self.encoder.input: x})

    #def decode(self, y, state):
    #    return self.sess.run(self.decoder.output, {self._latent_in: y, self.start_state: state})

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)




if __name__ == '__main__':
    import pylab as plt
    LOAD = True
    TRAIN = True
