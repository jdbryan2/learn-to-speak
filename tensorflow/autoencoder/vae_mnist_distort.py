
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.layers import conv1d, dense
from autoencoder import Autoencoder
from neural_network import NeuralNetwork
import tensorflow_utilities as tf_util

from tensorflow.examples.tutorials.mnist import input_data

def distortion(x):
    return np.fft.fftshift(np.abs(np.fft.fft(x)))

class VAE(Autoencoder):
    def __init__(self, latent_size, lr=1e-4, **kwargs):
        # input
        x = tf.placeholder(tf.float32, (None, 28*28))
        self.x = x
        y = tf.placeholder(tf.float32, (None, 28*28))
        self._y = y
        # encoder
        with tf.name_scope('encoder'):
            enc1 = dense(x, 50, activation=tf.nn.relu, name='enc1')
            enc2 = dense(enc1, 50, activation=tf.nn.relu, name='enc2')
            enc3 = dense(enc2, 50, activation=tf.nn.relu, name='enc3')
            #enc4 = dense(enc3, 50, activation=tf.nn.relu, name='enc4')

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
            
            #tx = tf_util.normalize_power(enc5, name='tx')
            #self.tx = tx
        # channel
        #with tf.name_scope('channel'):
            #rx = tf_util.awgn(tx, -snr, name='rx')
            #self.rx = rx


        # decoder
        with tf.name_scope('decoder'):
            #dec1 = conv1d(x, 4, 11, activation=tf.nn.relu, name='dec1')
            # NOTE: following line is tightly coupled to the architecture
            # dec1 is of size (None, 236, 4)
            dec1 = dense(self.rx, 50, activation=tf.nn.relu, name='dec1')
            dec2 = dense(dec1, 50, activation=tf.nn.relu, name='dec2')
            dec3 = dense(dec2, 50, activation=tf.nn.relu, name='dec3')
            #dec4 = dense(dec3, 50, activation=tf.nn.relu, name='dec4')

            x_out = dense(dec3, 28*28, activation=tf.nn.sigmoid, name='rx_out')
            self.x_out = x_out

        # quality metrics
        with tf.name_scope('metrics'):
            # change to MSE for accuracy, tx_power has no bearing
            self.accuracy = tf.reduce_mean(tf.squared_difference(y, x_out))#tf_util.accuracy(x, self.rx_bits, name='accuracy')
            #self.tx_power = tf_util.power(tx, name='tx_power')
        # loss
        with tf.name_scope('loss'):
            img_loss = tf.reduce_sum(tf.squared_difference(y, x_out), 1)
            latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
            loss = tf.reduce_mean(img_loss + latent_loss)
            #loss = tf_util.sigmoid_cross_entropy_loss( x, rx_logits, name='loss')
            self.loss = loss
        # optimizer
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        # initialize the base class
        #metrics = [self.loss, self.accuracy, self.tx_power]
        metrics = [self.loss, self.accuracy]
        encoder = NeuralNetwork(x, self.tx)
        self.encoder_std = NeuralNetwork(x, self.tx_std) # get the deviation on the encoder output
        decoder = NeuralNetwork(self.rx, self.x_out)
        Autoencoder.__init__(
            self, encoder, decoder, self.train_step, metrics=metrics, **kwargs
        )

    def encode_std(self, x):
        return self.sess.run(self.encoder_std.output, {self.encoder_std.input: x})

    def get_feed_dict(self, dset, n=None, batch_size=None):
        if n is None or batch_size is None:
            y = dset.data()
        else:
            y = dset.get_batch(n, batch_size)

        x = distortion(y)

        return {self.x: x, self._y:y}

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

class UnsupervisedDataset:
    def __init__(self, x):
        self._data = x

    def num_batches(self, batch_size):
        return int(np.floor(self._data.shape[0]/batch_size))

    def get_batch(self, batch_size, n=0):
        return self._data[n*batch_size:(n+1)*batch_size]

    def data(self):
        return self._data

class MNIST_Dataset:
    def __init__(self, directory, train=True):
        mnist = input_data.read_data_sets(directory)
        if train:
            self.mnist = mnist.train
        else:
            self.mnist = mnist.test

    def num_batches(self, batch_size):
        return int(np.floor(self.mnist.num_examples/batch_size))

    def num_points(self):
        return self.mnist.num_examples

    def get_batch(self, batch_size, n=0):
        batch, _  = self.mnist.next_batch(batch_size)
        return batch

    def data(self):
        batch, _  = self.mnist.next_batch(self.num_points())
        return batch
        #return self._data

if __name__ == '__main__':
    import pylab as plt
    TRAIN = True

    #log_dir = '/home/jacob/Projects/Data/vae/mnist-test'
    log_dir = '/home/jbryan/Data/vae-test'
    save_path = './trained/mnist_vae.ckpt'
    mnist_path = '/home/jbryan/mnist'

    #from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    model = VAE( 5, log_dir=log_dir, auto_logging=False, sess=session)

    session.run(tf.global_variables_initializer())

    #print('Training iteration #{0}'.format(n))
    d_train = MNIST_Dataset(mnist_path)
    d_val = MNIST_Dataset(mnist_path, train=False)
    if TRAIN:
        model.train(d_train, epochs=1, batch_size=50, d_val=d_val)
        model.save('./trained/mnist_vae.ckpt')
    else:
        model.load('./trained/mnist_vae.ckpt')



    tx = (0.5-np.random.random((5,5)))*2.
    img_out = model.decode(tx)
    img_in = distortion(img_out)
    rx = model.encode(img_out)
    rx_std = model.encode_std(img_out)

    error = np.abs(tx-rx)
    print(error)

    #for k in range(error.shape[0]):
    #    plt.figure()
    #    plt.plot(error[k])
    #    plt.plot(rx_std[k], 'r--')
    #    plt.plot(-1.*rx_std[k], 'r--')

    #plt.show()


    #img_in = d_val.get_batch(5)
    #encoded = model.encode(img_in)
    #img_out = model.decode(encoded)

    #re_encoded = model.encode(img_out)

    #error = np.abs(encoded-re_encoded)

    #for k in range(img_in.shape[0]):
    #    plt.figure()
    #    plt.imshow(img_in[k].reshape((28, 28)))
    #    plt.figure()
    #    plt.imshow(img_out[k].reshape((28, 28)))

    #    plt.figure()
    #    plt.plot(encoded[k])
    #    plt.plot(re_encoded[k])
    #    plt.plot(error[k])
    #    plt.show()


    

    #print encoded
    
