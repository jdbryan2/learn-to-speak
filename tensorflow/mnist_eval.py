

import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.vae import VAE, MNIST_Dataset, variable_summaries, null_distortion, blur_distortion

# Flags for saving time
LOAD = False  # load a previsou model at the location of load_path
#TRAIN = True # train the model for another round of epochs
BLUR = True # whether or not to use the blurring distortion
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

save_dir = './trained'

#log_dir = '/home/jacob/Projects/Data/vae/mnist-test'
#log_dir = '/home/jbryan/Data/vae-test'
load_path = save_dir+'/'+test_name+'.ckpt'

mnist_path = '/home/jacob/Projects/Data/MNIST_data'
#mnist_path = '/home/jbryan/mnist' # badger path

input_dim = 28*28
output_dim = 28*28
latent_size = 3



#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, output_dim, auto_logging=False, sess=session)

session.run(tf.global_variables_initializer())

#print('Training iteration #{0}'.format(n))
d_train = MNIST_Dataset(mnist_path, distortion)
d_val = MNIST_Dataset(mnist_path, distortion, train=False)


###
model.load(load_path)

num_points = 5000
tx = (0.5-np.random.random((num_points, latent_size)))*2.
#base = np.arange(-1., 1., 0.05)
#tx_0 = np.append(base, np.zeros(base.size*2))
#tx_1 = np.append(np.zeros(base.size), base)
#tx_1 = np.append(tx_1, np.zeros(base.size))
#tx_2 = np.append(np.zeros(base.size*2), base)

#print tx_0.size, tx_1.size, tx_2.size
#tx= np.array([tx_0, tx_1, tx_2]).T

img_out = model.decode(tx)
img_in = distortion(img_out)
rx = model.encode(img_out)
rx_std = model.encode_std(img_out)

error = np.abs(tx-rx)**2
error= np.sqrt(np.sum(error, axis=1))

std = np.exp(rx_std)**2
std = np.sqrt(np.sum(std, axis=1))

plt.scatter(error, std)
plt.xlabel('Error')
plt.ylabel('Standard Deviation')

ind = np.argsort(error)

#for k in range(5):
#    plt.figure()
#    mnist_show(img_out[ind[k]])

#plt.show()

#print(error)
print 'Average transmission error: ', np.mean(error)
print 'Error standard deviation: ', np.std(error)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
ind= np.where(error<0.2)
ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='b', marker='^')
ind= np.where((error<0.3)*(error > 0.2))
ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='g', marker='^')
#ind= np.where((error<0.4)* (error > 0.3))
#ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='y', marker='^')
#ind= np.where((error<0.5)*( error > 0.4))
#ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='orange', marker='^')
ind= np.where(error>1.)
ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='r', marker='o')
#ax.scatter(rx[:, 0], rx[:, 1], rx[:, 2], c='r', marker='o')#, s=np.mean(np.log(np.abs(rx_std)), axis=1))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()



#for k in range(tx.shape[0]):
#    plt.figure()
#    plt.plot(tx[k])
#    plt.plot(rx[k], 'r')
#    plt.plot(rx[k]+rx_std[k], 'r--')
#    plt.plot(rx[k]-rx_std[k], 'r--')
#    plt.figure()
#    mnist_show(img_out[k])
#    
#    plt.show()



#
#
#img_in, _ = d_val.get_batch(num_points)
#img_in = distortion(img_in)
#encoded = model.encode(img_in)
#img_out = model.decode(encoded)
#
#re_encoded = model.encode(img_out)
#
#error = np.abs(encoded-re_encoded)
#
## visualize some outputs
#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
## For each set of style and range settings, plot n random points in the box
## defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#ax.scatter(encoded[:, 0], encoded[:, 1], encoded[:, 2], c='b', marker='^')
#ax.scatter(re_encoded[:, 0], re_encoded[:, 1], re_encoded[:, 2], c='r', marker='o')
#
#ax.set_xlabel('X Label')
#ax.set_ylabel('Y Label')
#ax.set_zlabel('Z Label')
#
#plt.show()



#print encoded
