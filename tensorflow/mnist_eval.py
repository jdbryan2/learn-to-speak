

import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig
from genfigures.plot_functions import *
from helper_functions import *

import tensorflow as tf
from autoencoder.vae import VAE, MNIST_Dataset, variable_summaries, null_distortion, blur_distortion

# Flags for saving time
LOAD = False  # load a previsou model at the location of load_path
#TRAIN = True # train the model for another round of epochs
BLUR = False # whether or not to use the blurring distortion
EPOCHS = 50
EXAMPLES = 0

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
    fig_dir = 'figures/mnist_blur'
    data_path = 'tikz/mnist_blur'
else:
    test_name = 'mnist_null_vae'
    distortion = null_distortion # function pointer
    fig_dir = 'figures/mnist_null'
    data_path = 'tikz/mnist_null'

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

# Plot input and output images
##############################
num_points = 1000
img_in, clean_img, label_num  = d_val.get_labeled_batch(0,num_points)

# encode and decode images to look at quality
rx = model.encode(img_in)
img_out = model.decode(rx)

if EXAMPLES == 0:
    for k in range(10):
        mnist_examples(label_num, img_in, img_out, select=k)
        #plt.show()
        tikz_save(fig_dir+'/examples_'+str(k)+'.tikz', data_path='tikz/mnist_null')
        #plt.show()
        plt.close()


exit()
num_points = 5000
#tx = (0.5-np.random.random((num_points, latent_size)))*2.
tx = np.random.normal(0, 1, (num_points, latent_size))

img_out = model.decode(tx)
img_in = distortion(img_out)
rx = model.encode(img_out)
rx_std = model.encode_std(img_out)

std = np.exp(2*rx_std)
std = np.sqrt(np.sum(std, axis=1))

#plt.scatter(error, std)
#plt.xlabel('Error')
#plt.ylabel('Standard Deviation')
#
#ind = np.argsort(error)

cap, MSE, power = channel_capacity(tx, rx)
cap = channel_capacity2(tx, rx)

print "MSE: ", MSE
print "Power: ", power
print "Channel Cap: ", cap

exit()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if BLUR:
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
else:
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    ind= np.where(error<0.05)
    ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='b', marker='^')
    ind= np.where((error<0.1)*(error > 0.05))
    ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='g', marker='^')
    #ind= np.where((error<0.4)* (error > 0.3))
    #ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='y', marker='^')
    #ind= np.where((error<0.5)*( error > 0.4))
    #ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='orange', marker='^')
    ind= np.where(error>0.2)
    ax.scatter(tx[ind, 0], tx[ind, 1], tx[ind, 2], c='r', marker='o')
    #ax.scatter(rx[:, 0], rx[:, 1], rx[:, 2], c='r', marker='o')#, s=np.mean(np.log(np.abs(rx_std)), axis=1))

    #ax.scatter(tx[:, 0], tx[:, 1], tx[:, 2], c='b', marker='o')
    #ax.scatter(rx[:, 0], rx[:, 1], rx[:, 2], c='r', marker='^')#, s=np.mean(np.log(np.abs(rx_std)), axis=1))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

if BLUR:
    ind, = np.where(error<0.2)
    np.random.shuffle(ind)
    for k in range(10):
        #mnist_show(img_out[ind[k], :])
        plt.imshow(np.append(img_out[ind[k]].reshape((28,28)), img_in[ind[k]].reshape((28,28)), axis=1),
                   interpolation='none')
        tikz_save('/home/jacob/Desktop/mnist_figures/low_error_%i.tikz'%k)
        plt.close()
        #plt.show()
    
    ind, = np.where(error>1.)
    np.random.shuffle(ind)
    for k in range(10):
        #mnist_show(img_out[ind[k], :])
        plt.imshow(np.append(img_out[ind[k]].reshape((28,28)), img_in[ind[k]].reshape((28,28)), axis=1),
                   interpolation='none')
        tikz_save('/home/jacob/Desktop/mnist_figures/high_error_%i.tikz'%k)
        plt.close()
else:

    ind = np.argsort(error)
    for k in range(10):
        #mnist_show(img_out[ind[k], :])
        plt.imshow(img_out[ind[k]].reshape((28,28)), interpolation='none')
        tikz_save('/home/jacob/Desktop/mnist_figures/low_error_%i.tikz'%k)
        plt.show()

        plt.imshow(img_out[ind[-k]].reshape((28,28)), interpolation='none')
        tikz_save('/home/jacob/Desktop/mnist_figures/high_error_%i.tikz'%k)
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
