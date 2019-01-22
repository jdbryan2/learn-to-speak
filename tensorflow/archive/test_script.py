
import numpy as np
import pylab as plt

import tensorflow as tf
import tensorflow_utilities as tf_util
from tensorflow.examples.tutorials.mnist import input_data
from vae_mnist_distort import VAE, distortion, MNIST_Dataset

LOAD = False
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
d_train = MNIST_Dataset(mnist_path)
d_val = MNIST_Dataset(mnist_path, train=False)


x = model.get_feed_dict(d_train, 10)

keys = x.keys()

y0 = x[keys[0]]
y1 = x[keys[1]]

for k in range(y0.shape[0]):
    plt.figure()
    plt.imshow(y0[k].reshape((28,28)))
    plt.figure()
    plt.imshow(y1[k].reshape((28,28)))
    plt.show()


