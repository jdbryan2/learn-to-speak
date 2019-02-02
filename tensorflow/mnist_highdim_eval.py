

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
BLUR = True # whether or not to use the blurring distortion
EXAMPLES = 2  
EPOCHS = 50



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
unused_dims = np.exp(np.mean(rx_std, axis=0))>0.9

# encode and decode images to look at quality
rx = model.encode(img_in)
img_out = model.decode(rx)

if EXAMPLES == 0:
    for k in range(10):
        mnist_examples(label_num, img_in, img_out, select=k)
        plt.show()
        #tikz_save(fig_dir+'/examples_'+str(k))
        #plt.show()
        #plt.close()

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
        #tikz_save(fig_dir+'/sweep_'+str(k))
        plt.show()

    #print np.sum(0.5*np.log2(1+1/err[dims]))
        #tikz_save(fig_dir+'/examples_'+str(k))
        #plt.close()







#tx = (0.5-np.random.random((num_points, latent_size)))*2.
tx = np.random.normal(0, 1, (num_points, latent_size))
tx[:, unused_dims] = 0.

img_out = model.decode(tx)
img_in = distortion(img_out)
rx = model.encode(img_in)
rx_std = model.encode_std(img_in)

#remove unused dimensions from analysis
rx = rx[:, ~unused_dims]
rx_std = rx_std[:, ~unused_dims]
tx = tx[:, ~unused_dims]
latent_size = latent_size - np.sum(unused_dims)

#for k in range(latent_size):
#    plt.figure()
#    plt.scatter(np.arange(latent_size), tx[k], c='blue', marker='o')
#    plt.scatter(np.arange(latent_size), rx[k], c='red', marker='^')
#    plt.show()

cap, MSE, power = channel_capacity(tx, rx)
cap= channel_capacity2(tx, rx)
#square_error = np.sum(np.abs(tx-rx)**2, axis=1)
#power = np.mean(np.sum(np.abs(tx)**2, axis=1))
#MSE= np.mean(square_error)

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
#c = np.sum(0.5*np.log2(1+latent_size/MSE))
#print 'Estimated Channel Capacity: ', c
#c2 = np.sum(0.5*np.log2(1+power/MSE))
print 'Estimated Channel Capacity (power): ', cap
print 'Estimated TX Power: ', power


#plt.plot(error[ind])
#plt.show()



dim_error = np.abs(tx-rx)**2
mean_error = np.mean(dim_error, axis=0)

cov_tx = np.cov(tx.T)
cov_rx = np.cov(rx.T)
cov_error = np.cov((tx-rx).T)

Kt = np.linalg.det(cov_tx)
Kr = np.linalg.det(cov_rx)
Ke = np.linalg.det(cov_error)
plt.imshow(cov_error)
plt.show()
Ht = 0.5*np.log2((2*np.pi*np.exp(1))**latent_size*Kt)
Hr = 0.5*np.log2((2*np.pi*np.exp(1))**latent_size*Kr)
He = 0.5*np.log2((2*np.pi*np.exp(1))**latent_size*Ke)

dim_var = np.exp(rx_std)
mean_var = np.mean(dim_var, axis=0)

#for k in range(dim_error.shape[1]):
#    plt.figure()
#    plt.scatter(dim_error[:, k], dim_var[:, k])
#plt.show()



