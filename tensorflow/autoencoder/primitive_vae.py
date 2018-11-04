
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
            # note: epsilon is shaped based on zeroth dim of enc4 so that it won't break if 
            #       enc4 is made convolutional
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
            self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd))
            
            #self.tx_power = tf_util.power(tx, name='tx_power')
        # loss
        with tf.name_scope('loss'):
            #art_loss = tf.reduce_sum(tf.squared_difference(x_out, self.target), 1)
            art_loss = tf.losses.absolute_difference(x_out, self.target)
            latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            loss = tf.reduce_mean(art_loss + latent_size/input_dim*latent_loss)
            #loss = tf.reduce_mean(img_loss)
            #loss = tf_util.sigmoid_cross_entropy_loss( x, rx_logits, name='loss')
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


def reshape_data(data,history):
    max_len = int(np.floor(data.shape[0]/history)*history)
    return data[:max_len, :].reshape((-1, data.shape[1]*history))


def LoadData(directory='speech_io', outputs_name='mfcc', inputs_name='action_segs', states_name='state_segs', shuffle=True):

    ind_list = np.array(get_index_list(directory=directory, base_name='data'))

    ind_list = np.sort(ind_list)

    outputs = np.array([])
    inputs = np.array([])
    states = np.array([])
    print "Loading Data..."
    for k in ind_list:
        data = np.load(directory+"/data%i.npz"%k)

        if outputs.size:
            outputs = np.append(outputs, data[outputs_name], axis=0)
            inputs = np.append(inputs, data[inputs_name][:, :, -1], axis=0)
            states = np.append(states, data[states_name][:, :, -1], axis=0)
            #spec = np.append(spec, data['mel_spectrum'], axis=0)
        else:
            outputs = data[outputs_name]
            inputs = data[inputs_name][:, :, -1]
            states = data[states_name][:, :, -1]
            #spec = data['mel_spectrum']

    #plt.imshow(spec.T, aspect=3*20)
    #plt.show()
    remove = ~np.isnan(outputs)
    outputs = outputs[remove].reshape((-1, 13))
    remove.reshape((-1, 13))
    inputs = inputs[remove[:, 0], :]
    states = states[remove[:, 0], :]
    print remove.shape

    E_m = np.mean(outputs[:, 0])
    E_sd = np.std(outputs[:, 0])
    print E_m, E_sd
    m_count = np.sum(outputs[:, 0]>E_m)
    sd_count = np.sum(outputs[:, 0]>E_m+E_sd)
    count = outputs.shape[0]

    print 1.*m_count/count, 1.*sd_count/count

    #MultiPlotTraces(actions.T, np.arange(actions.shape[1]), actions.shape[0], 80)
    #plt.show()

    if shuffle:
        new_index = np.arange(0, inputs.shape[0]).astype('int')
        np.random.shuffle(new_index)

        return inputs[new_index, :], outputs[new_index, :], states[new_index,:]
    else: 
        return inputs, outputs, states

class PyraatDataset2:
    def __init__(self, actions, features):
        self.actions = actions
        self.features = features

    def num_batches(self, batch_size):
        return int(np.floor(self.features.shape[0]/batch_size))

    def parameter_sizes(self):
        return self.features.shape[1], self.actions.shape[1]+self.features.shape[1]

    def get_batch(self, n=0, batch_size=50):
        start = n*batch_size
        stop = (n+1)*batch_size
        return self.features[start:stop, :], np.append(self.actions[start:stop, :], self.features[start:stop, :],
                                                       axis=1)
        #return self._input[start:stop, :], self._output[start:stop, :]

    def data(self):
        start = 0
        stop = self.actions.shape[0]
        #return self.features[start:stop, :], self.actions[start:stop, :]
        return self.features[start:stop, :], np.append(self.actions[start:stop, :], self.features[start:stop, :],
                                                       axis=1)
        #return self._input[start:stop, :], self._output[start:stop, :]

class PyraatDataset:
    def __init__(self, actions, features):
        self.actions = actions
        self.features = features

    def num_batches(self, batch_size):
        return int(np.floor(self.features.shape[0]/batch_size))

    def parameter_sizes(self):
        return self.features.shape[1], self.actions.shape[1]

    def get_batch(self, n=0, batch_size=50):
        start = n*batch_size
        stop = (n+1)*batch_size
        return self.features[start:stop, :], self.actions[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]

    def data(self):
        start = 0
        stop = self.actions.shape[0]
        return self.features[start:stop, :], self.actions[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]
        

if __name__ == '__main__':
    import pylab as plt
    LOAD = True
    TRAIN = True


