

import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig
from genfigures.plot_functions import *

import tensorflow as tf
from autoencoder.vae import VAE, MNIST_Dataset, variable_summaries, null_distortion, blur_distortion

# Flags for saving time
LOAD = False  # load a previsou model at the location of load_path
#TRAIN = True # train the model for another round of epochs
BLUR = False # whether or not to use the blurring distortion
EPOCHS = 50

# helper functions

# wrapper for showing mnist vectors as images
def mnist_show(x):
    plt.imshow(x.reshape((28, 28)))

def scatter3D(data, color='b', marker='o'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, marker=marker)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
# blur


if BLUR:
    test_name = 'mnist_blur_vae'
    distortion = blur_distortion  # function pointer
else:
    test_name = 'mnist_null_vae'
    distortion = null_distortion # function pointer

#save_dir = './trained/mnist_hd'
save_dir = './trained'

#log_dir = '/home/jacob/Projects/Data/vae/mnist-test'
#log_dir = '/home/jbryan/Data/vae-test'
load_path = save_dir+'/'+test_name+'.ckpt'

mnist_path = '/home/jacob/Projects/Data/MNIST_data'
#mnist_path = '/home/jbryan/mnist' # badger path

input_dim = 28*28
output_dim = 28*28
latent_size = 3

colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'gray', 'cyan', 'saddlebrown', 'fuchsia']


#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, output_dim, auto_logging=False, sess=session)

session.run(tf.global_variables_initializer())

#print('Training iteration #{0}'.format(n))
d_train = MNIST_Dataset(mnist_path, distortion)
d_val = MNIST_Dataset(mnist_path, distortion, train=False)


###
model.load(load_path)

num_points = 1000

img_in, clean_img, label_num  = d_train.get_labeled_batch(0,num_points)
#_, label_num = np.where(label)
ind = np.argsort(label_num)
label_num = label_num[ind]
img_in = img_in[ind] # group by label

rx = model.encode(img_in)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for k in range(10):
    ind = np.where(label_num == k)
    ax.scatter(rx[ind, 0], rx[ind, 1], rx[ind, 2], c=colors[k], marker='o')#, s=np.mean(np.log(np.abs(rx_std)), axis=1))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

