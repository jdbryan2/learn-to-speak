
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
    def __init__(self, input_dim, latent_size, output_dim, lr=1e-4, netwidth=50, **kwargs):
        # input
        x = tf.placeholder(tf.float32, (None, input_dim))
        self.x = x
        self.target = tf.placeholder(tf.float32, (None, output_dim))
        #self.y = y
        # encoder
        with tf.name_scope('encoder'):
            enc1 = dense(x, netwidth, activation=tf.nn.relu, name='enc1')
            enc2 = dense(enc1, netwidth, activation=tf.nn.relu, name='enc2')
            enc3 = dense(enc2, netwidth, activation=tf.nn.relu, name='enc3')
            #enc4 = dense(enc3, netwidth, activation=tf.nn.relu, name='enc4')

            # variational layers
            # note: epsilon is shaped based on zeroth dim of enc4 so that it won't break if 
            #       enc4 is made convolutional
            mn = dense(enc3, units=latent_size)
            sd       = 0.5 * tf.layers.dense(enc3, units=latent_size)            
            epsilon = tf.random_normal(tf.stack([tf.shape(enc3)[0], latent_size])) 

            samples  = mn + tf.multiply(epsilon, tf.exp(sd))
            self.tx = mn
            self.tx_std = sd
            self.rx = samples
            
        # decoder
        with tf.name_scope('decoder'):
            # NOTE: following line is tightly coupled to the architecture
            # dec1 is of size (None, 236, 4)
            dec1 = dense(self.rx, netwidth, activation=tf.nn.relu, name='dec1')
            dec2 = dense(dec1, netwidth, activation=tf.nn.relu, name='dec2')
            dec3 = dense(dec2, netwidth, activation=tf.nn.relu, name='dec3')
            #dec4 = dense(dec3, 50, activation=tf.nn.relu, name='dec4')

            x_out = dense(dec3, output_dim, activation=tf.nn.sigmoid, name='rx_out')
            self.x_out = x_out

        # quality metrics
        with tf.name_scope('metrics'):
            # change to MSE for accuracy, tx_power has no bearing
            self.accuracy = tf.reduce_mean(tf.squared_difference(x_out, self.target))#tf_util.accuracy(x, self.rx_bits, name='accuracy')
            self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1))
            #self.tx_power = tf_util.power(tx, name='tx_power')
        # loss
        with tf.name_scope('loss'):
            img_loss = tf.reduce_sum(tf.squared_difference(x_out, self.target), 1)
            latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            #loss = tf.reduce_mean(img_loss + 0.0128*input_dim/latent_size*latent_loss)
            loss = tf.reduce_sum(img_loss + latent_loss)
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


class MNIST_Dataset:
    def __init__(self, directory, distortion_callback, train=True):

        # save a distortion function as the callback
        self.distort = distortion_callback

        mnist = input_data.read_data_sets(directory)
        if train:
            self.mnist = mnist.train
        else:
            self.mnist = mnist.test

    def num_batches(self, batch_size):
        return int(np.floor(self.mnist.num_examples/batch_size))

    def num_points(self):
        return self.mnist.num_examples

    def get_batch(self, n=0, batch_size=50):
        batch, _  = self.mnist.next_batch(batch_size)
        return self.distort(batch), batch

    def get_labeled_batch(self, n=0, batch_size=50):
        batch, labels  = self.mnist.next_batch(batch_size)
        return self.distort(batch), batch, labels

    def data(self):
        batch, _  = self.mnist.next_batch(self.num_points())
        return self.distort(batch), batch

        #return self._data

def null_distortion(x):
    return np.copy(x)

def blur_distortion(x):
    blur_size = 5
    return sig.lfilter(np.ones(blur_size), [1], x)/blur_size

if __name__ == '__main__':
    import pylab as plt
    LOAD = True
    TRAIN = True

    log_dir = '/home/jacob/Projects/Data/vae/mnist-test'
    #log_dir = '/home/jbryan/Data/vae-test'
    save_path = './garbage/mnist_vae.ckpt'
    load_path = './garbage/mnist_vae.ckpt'
    mnist_path = '/home/jacob/Projects/Data/MNIST_data'
    #mnist_path = '/home/jbryan/mnist'

    #from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    model = VAE( 5, log_dir=log_dir, auto_logging=False, sess=session)

    session.run(tf.global_variables_initializer())

    #print('Training iteration #{0}'.format(n))
    d_train = MNIST_Dataset(mnist_path, null_distortion)
    d_val = MNIST_Dataset(mnist_path, nulle_distortion, train=False)
    if LOAD:
        model.load(load_path)
    if TRAIN:
        model.train(d_train, epochs=20, batch_size=50, d_val=d_val)
        model.save(save_path)



    tx = (0.5-np.random.random((5,5)))*2.
    img_out = model.decode(tx)
    img_in = distortion(img_out)
    rx = model.encode(img_out)
    rx_std = model.encode_std(img_out)

    error = np.abs(tx-rx)
    print(error)

    for k in range(error.shape[0]):
        plt.figure()
        plt.plot(error[k])
        plt.plot(rx_std[k], 'r--')
        plt.plot(-1.*rx_std[k], 'r--')

    plt.show()


    img_in = d_val.get_batch(5)
    img_in = distortion(img_in)
    encoded = model.encode(img_in)
    img_out = model.decode(encoded)

    re_encoded = model.encode(img_out)

    error = np.abs(encoded-re_encoded)

    for k in range(img_in.shape[0]):
        plt.figure()
        plt.imshow(img_in[k].reshape((28, 28)))
        plt.figure()
        plt.imshow(img_out[k].reshape((28, 28)))

        plt.figure()
        plt.plot(encoded[k])
        plt.plot(re_encoded[k])
        plt.plot(error[k])
        plt.show()


    

    #print encoded
    
