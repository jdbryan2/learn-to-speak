

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
BLUR = True # whether or not to use the blurring distortion
EXAMPLES = 1
EPOCHS = 50


# helper functions

# wrapper for showing mnist vectors as images
def mnist_show(x, y):
    plt.imshow(np.append(x.reshape((28,28)), y.reshape((28,28)), axis=0), interpolation='none')
    #plt.imshow(x.reshape((28, 28)))

def mnist_sequence(x, y):
    if x.shape[0] != y.shape[0]:
        print "Sequences of images must be same size"
        return 0

    
    if x.shape[0] > 1:
        image = np.append(x[0].reshape((28,28)), y[0].reshape((28,28)), axis=0)
        for k in range(1, x.shape[0]):
            image = np.append(image, np.append(x[k].reshape((28,28)), y[k].reshape((28,28)), axis=0), axis=1)

    else:
        image = np.append(x.reshape((28,28)), y.reshape((28,28)), axis=0)

    return plt.imshow(image, interpolation='none')


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
    test_name = 'mnist_blur_hd'
    distortion = blur_distortion  # function pointer
else:
    test_name = 'mnist_null_hd'
    distortion = null_distortion # function pointer

fig_dir = './figures/'+test_name

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

#log_dir = '/home/jacob/Projects/Data/vae/mnist-test'
#log_dir = '/home/jbryan/Data/vae-test'
save_dir = './trained/mnist_hd'
load_path = save_dir+'/'+test_name+'.ckpt'

mnist_path = '/home/jacob/Projects/Data/MNIST_data'
#mnist_path = '/home/jbryan/mnist' # badger path

input_dim = 28*28
output_dim = 28*28
latent_size = 10 



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

# identify unused dimensions
img_in, clean_img, label_num  = d_val.get_labeled_batch(0,num_points)

rx_std = model.encode_std(img_in)
unused_dims = np.exp(np.mean(rx_std, axis=0))>0.99

# encode and decode images to look at quality
rx = model.encode(img_in)
img_out = model.decode(rx)

if EXAMPLES == 0:
    for k in range(10):
        ind, = np.where(label_num==k)

        #plt.figure()
        img_handle = mnist_sequence(img_in[ind[1:10]], img_out[ind[1:10]])
        img_handle.set_cmap('Greys')
        plt.axis('off')
        tikz_save(fig_dir+'/examples_'+str(k))
        #plt.show()
        plt.close()

#exit()

if EXAMPLES == 1:
    
    dims = np.where(~unused_dims)[0]
    samples = np.arange(-1, 1.01, 0.22)
    err = np.zeros(latent_size)
    for k in dims:
        tx = 0.*np.ones((samples.size, latent_size))
        tx[:, k] = samples

        img_out = model.decode(tx)

        img_in = distortion(img_out)
        rx = model.encode(img_in)
        img_out2 = model.decode(rx)

        err[k] = np.mean(np.sum(np.abs(tx-rx)**2, axis=1))
        #print np.mean(err)

        plt.figure()
        img_handle = mnist_sequence(img_out, img_out2)
        img_handle.set_cmap('Greys')
        plt.axis('off')
        tikz_save(fig_dir+'/sweep_'+str(k))
        plt.show()

    print np.sum(0.5*np.log2(1+1/err[dims]))
        #tikz_save(fig_dir+'/examples_'+str(k))
        #plt.close()







tx = (0.5-np.random.random((num_points, latent_size)))*2.
tx[:, unused_dims] = 0.

#base = np.arange(-1., 1., 0.05)
#tx_0 = np.append(base, np.zeros(base.size*2))
#tx_1 = np.append(np.zeros(base.size), base)
#tx_1 = np.append(tx_1, np.zeros(base.size))
#tx_2 = np.append(np.zeros(base.size*2), base)
#
##print tx_0.size, tx_1.size, tx_2.size
#tx= np.array([tx_0, tx_1, tx_2]).T

img_out = model.decode(tx)
img_in = distortion(img_out)
rx = model.encode(img_in)
rx_std = model.encode_std(img_in)

rx = rx[:, ~unused_dims]
rx_std = rx_std[:, ~unused_dims]
tx = tx[:, ~unused_dims]
latent_size = latent_size - np.sum(unused_dims)

#for k in range(latent_size):
#    plt.figure()
#    plt.scatter(np.arange(latent_size), tx[k], c='blue', marker='o')
#    plt.scatter(np.arange(latent_size), rx[k], c='red', marker='^')
#    plt.show()

square_error = np.sum(np.abs(tx-rx)**2, axis=1)
MSE= np.mean(square_error)

var = np.sum(np.exp(2*rx_std)**2, axis=1)
var = np.mean(var)

#plt.scatter(error, std)
#plt.xlabel('Error')
#plt.ylabel('Standard Deviation')
#plt.show()

#ind = np.argsort(error)

#for k in range(5):
#    plt.figure()
#    mnist_show(img_out[ind[k]])

#plt.show()

#print(error)
print 'Transmission MSE: ', MSE
#print 'Error standard deviation: ', np.std(error)
print 'Mean RX Variance: ', var
c = np.sum(0.5*np.log2(1+latent_size/MSE))
print 'Estimated Channel Capacity: ', c

#plt.plot(error[ind])
#plt.show()



dim_error = np.abs(tx-rx)**2
mean_error = np.mean(dim_error, axis=0)

cov_error = np.cov((tx-rx).T)
plt.imshow(cov_error)
plt.show()


dim_var = np.exp(2*rx_std)**2
mean_var = np.mean(dim_var, axis=0)

#for k in range(dim_error.shape[1]):
#    plt.figure()
#    plt.scatter(dim_error[:, k], dim_var[:, k])
#plt.show()



