
import os
import numpy as np
import scipy.signal as sig
import tensorflow as tf
from tensorflow.python.layers.layers import conv1d, dense
from autoencoder import Autoencoder
from neural_network import NeuralNetwork
import tensorflow_utilities as tf_util

from tensorflow.examples.tutorials.mnist import input_data

# The basic idea here is that the autoencoder is learning a function approxation 
# rather than the identity function. 
# This allows it to learn an inverse transform that can be used as a channel for transmitting

class VAE(Autoencoder):
    #def __init__(self, input_dim, latent_size, state_dim, output_dim, inner_width=50, lr=1e-4, **kwargs):

    def __init__(self, input_dim, latent_size, output_dim, inner_width=50, lr=1e-4, **kwargs):
        # input
        x = tf.placeholder(tf.float32, (None, input_dim))
        self.x = x
        self.target = tf.placeholder(tf.float32, (None, output_dim))

        activation = tf.nn.relu
        #self.start_state = tf.placeholder(tf.float32, (None, state_dim))
        #self.y = y
        # encoder
        with tf.name_scope('encoder'):
            enc1 = dense(x, inner_width, activation=activation, name='enc1')
            enc2 = dense(enc1, inner_width, activation=activation, name='enc2')
            enc3 = dense(enc2, inner_width, activation=activation, name='enc3')
            #enc4 = dense(enc3, inner_width, activation=tf.nn.relu, name='enc4')

            # variational layers
            # note: epsilon is shaped based on zeroth dim of enc4 so that it won't break if 
            #       enc4 is made convolutional
            #mn = dense(enc3, units=latent_size)
            #sd       = 0.5 * tf.layers.dense(enc3, units=latent_size)            
            #epsilon = tf.random_normal(tf.stack([tf.shape(enc3)[0], latent_size])) 

            #samples  = mn + tf.multiply(epsilon, tf.exp(sd))
            #self.tx = mn
            #self.tx_std = sd
            self.tx = enc3
            self.rx= enc3
            #self._latent_in = samples
            #self.rx = tf.concat((samples, self.start_state), axis=1) # starting state feeds into the latent space
            
        # decoder
        with tf.name_scope('decoder'):
            # NOTE: following line is tightly coupled to the architecture
            # dec1 is of size (None, 236, 4)
            dec1 = dense(self.rx, inner_width, activation=activation, name='dec1')
            dec2 = dense(dec1, inner_width, activation=activation, name='dec2')
            dec3 = dense(dec2, inner_width, activation=activation, name='dec3')
            #dec4 = dense(dec3, inner_width, activation=tf.nn.relu, name='dec4')

            x_out = dense(dec3, output_dim, activation=activation, name='rx_out')
            #x_out = dense(dec3, output_dim, activation=tf.nn.relu, name='rx_out')
            self.x_out = x_out

        # quality metrics
        with tf.name_scope('metrics'):
            # change to MSE for accuracy, tx_power has no bearing
            self.accuracy = tf.reduce_mean(tf.squared_difference(x_out, self.target))#tf_util.accuracy(x, self.rx_bits, name='accuracy')
            #self.latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd))
            #self.tx_power = tf_util.power(tx, name='tx_power')
        # loss
        with tf.name_scope('loss'):
            #img_loss = tf.reduce_sum(tf.squared_difference(x_out, self.target), 1)
            #loss = tf.nn.l2_loss(x_out - self.target)
            loss = tf.losses.absolute_difference(x_out, self.target)
            #latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            #loss = tf.reduce_mean(img_loss + latent_loss)
            #loss = tf.reduce_mean(img_loss)
            #loss = tf_util.sigmoid_cross_entropy_loss( x, rx_logits, name='loss')
            self.loss = loss
        # optimizer
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(loss)

        # initialize the base class
        #metrics = [self.loss, self.accuracy, self.tx_power]
        metrics = [self.loss, self.accuracy] #, self.latent_loss]
        encoder = NeuralNetwork(x, self.tx)
        #self.encoder_std = NeuralNetwork(x, self.tx_std) # get the deviation on the encoder output
        decoder = NeuralNetwork(self.rx, self.x_out)
        Autoencoder.__init__(self, encoder, decoder, self.train_step, metrics=metrics, **kwargs)

    #def encode_std(self, x):
    #    return self.sess.run(self.encoder_std.output, {self.encoder_std.input: x})

    # note: self.target must get defined in the inheriting class
    def get_feed_dict(self, dset, n=None, batch_size=None):
        if n is None or batch_size is None:
            x, target = dset.data()
        else:
            x, target = dset.get_batch(n, batch_size)
        #return {self.x: x, self.target: target, self.start_state: state}
        return {self.x: x, self.target: target}

    #def encode(self, x):
    #    return self.sess.run(self.encoder.output, {self.encoder.input: x})

    #def decode(self, y, state):
    #    return self.sess.run(self.decoder.output, {self.decoder.input: y})
    #def decode(self, y, state):
        #return self.sess.run(self.decoder.output, {self._latent_in: y, self.start_state: state})

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

class Dynamical_Dataset:
    def __init__(self, directory, history_length, train=True):
        self.directory = directory
        self.history_length = history_length

        # pull indeces from the filenames
        index_list = []  # using a list for simplicity
        for filename in os.listdir(directory):
            if filename.startswith('batch_') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, str(filename))))

        # sort numerically and load files in order
        self.index_list = np.array(sorted(index_list))
        
        fname=directory+"/batch_%i.npz"
        self.action = np.array([])
        self.state = np.array([])
        self.obs = np.array([])
        for ind in self.index_list:
            data = np.load(fname%ind)
            #print fname%ind
            #print data.keys()
            #print self.action.shape, self.state.shape, self.obs.shape
            action = reshape_data(data['action'], history_length)
            state = reshape_data(data['state'], history_length)
            obs = reshape_data(data['observation'], history_length)
            if self.action.size == 0:
                self.action = action
                self.state = state[:, :4] # only keep the starting state
                self.obs = obs
            else:
                self.action = np.append(self.action, action, axis=0)
                self.state = np.append(self.state, state[:, :4], axis=0)
                self.obs = np.append(self.obs, obs, axis=0)

        # need to add a step where the data is shuffled 
        new_index = np.arange(0, self.action.shape[0]).astype('int')
        np.random.shuffle(new_index)

        self.action[:, 1] = 0
        #self.action = self.action[new_index, :]
        #self.state = self.state[new_index, :]
        #self.obs = self.obs[new_index, :]

        # normalize
        self.action_mean = np.mean(self.action, axis=0)
        #self.action_std = np.std(self.action, axis=0)
        self.action_std = np.max(self.action) - np.min(self.action)
        self.obs_mean = np.mean(self.obs, axis=0)
        #self.obs_std = np.std(self.obs, axis=0)
        self.obs_std = np.max(self.obs, axis=0) - np.min(self.obs,axis=0)
        self.action = (self.action-np.mean(self.action, axis=0))/self.action_std
        self.state = (self.state-np.mean(self.state, axis=0))/np.std(self.state, axis=0)
        self.obs = (self.obs-np.mean(self.obs, axis=0))/self.obs_std


    def num_batches(self, batch_size):
        return int(np.floor(self.action.shape[0]/batch_size))

    #def num_points(self):
        #return self.mnist.num_examples

    def parameter_sizes(self):
        return self.obs.shape[1], self.action.shape[1], self.state.shape[1]

    def get_batch(self, n=0, batch_size=50):
        start = n*batch_size
        stop = (n+1)*batch_size
        return self.obs[start:stop, :], self.action[start:stop, :], self.state[start:stop, :] 

    def data(self):
        start = 0
        stop = self.action.shape[0]
        return self.obs[start:stop, :], self.action[start:stop, :], self.state[start:stop, :] 


        

if __name__ == '__main__':
    import pylab as plt
    LOAD = True
    TRAIN = True


