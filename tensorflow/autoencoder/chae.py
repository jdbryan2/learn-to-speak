import numpy as np
import tensorflow as tf
from tensorflow.python.layers.layers import conv1d, dense
from autoencoder import Autoencoder
from neural_network import NeuralNetwork
import tensorflow_utilities as tf_util


class ChAE(Autoencoder):
    def __init__(self, snr, lr=1e-4, **kwargs):
        # input
        x = tf.placeholder(tf.float32, (None, 128))
        self.x = x
        # encoder
        with tf.name_scope('encoder'):
            enc1 = dense(x, 256, activation=tf.nn.relu, name='enc1')
            enc2 = dense(enc1, 256, activation=tf.nn.relu, name='enc2')
            enc3 = dense(enc2, 256, activation=tf.nn.relu, name='enc3')
            enc4 = dense(enc3, 256, activation=tf.nn.relu, name='enc4')
            enc5 = tf.reshape(enc4, (-1, 256, 1), name='rs1')
            enc5 = conv1d(enc5, 2, 11, activation=tf.identity, name='enc5')
            tx = tf_util.normalize_power(enc5, name='tx')
            self.tx = tx
        # channel
        with tf.name_scope('channel'):
            rx = tf_util.awgn(tx, -snr, name='rx')
            self.rx = rx
        # decoder
        with tf.name_scope('decoder'):
            dec1 = conv1d(rx, 4, 11, activation=tf.nn.relu, name='dec1')
            # NOTE: following line is tightly coupled to the architecture
            # dec1 is of size (None, 236, 4)
            dec1 = tf.reshape(dec1, (-1, 236*4), 'rs2')

            dec2 = dense(dec1, 256, activation=tf.nn.relu, name='dec2')
            dec3 = dense(dec2, 256, activation=tf.nn.relu, name='dec3')
            dec4 = dense(dec3, 256, activation=tf.nn.relu, name='dec4')
            dec5 = dense(dec4, 256, activation=tf.nn.relu, name='dec5')
            rx_logits = dense(
                dec5, 128, activation=tf.nn.sigmoid, name='rx_logits')
            self.rx_bits = tf.cast(
                tf.greater(rx_logits, 0.5, name='rx_bits'), tf.float32)
        # quality metrics
        with tf.name_scope('metrics'):
            self.accuracy = tf_util.accuracy(x, self.rx_bits, name='accuracy')
            self.tx_power = tf_util.power(tx, name='tx_power')
        # loss
        with tf.name_scope('loss'):
            loss = tf_util.sigmoid_cross_entropy_loss(
                x, rx_logits, name='loss')
            self.loss = loss
        # optimizer
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(lr).minimize(loss)
        # initialize the base class
        metrics = [self.loss, self.accuracy, self.tx_power]
        encoder = NeuralNetwork(x, tx)
        decoder = NeuralNetwork(rx, self.rx_bits)
        Autoencoder.__init__(
            self, encoder, decoder, self.train_step, metrics=metrics, **kwargs
        )


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

if __name__ == '__main__':
    #from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
    session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    model = ChAE(
        5, log_dir='/home/jacob/Projects/sandbox/tensorflow/test-1', auto_logging=False,
        sess=session
    )
    session.run(tf.global_variables_initializer())
    for n in range(1):
        print('Training iteration #{0}'.format(n))
        d_train = UnsupervisedDataset(np.random.randint(0, 2, (100000, 128)))
        d_val = UnsupervisedDataset(np.random.randint(0, 2, (1024, 128)))
        model.train(d_train, epochs=2, batch_size=128, d_val=d_val)

    model.save('./')
