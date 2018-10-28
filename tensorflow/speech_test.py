
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.pyraat_vae import VAE, PyraatDataset2, LoadData, variable_summaries

from genfigures.plot_functions import *

actions, features = LoadData(directory='speech_io', inputs='art_segs', outputs='mfcc')

#mean_f = np.mean(features, axis=0)
min_f = np.min(features, axis=0)
max_f = np.max(features, axis=0)
features = (features-min_f)/(max_f - min_f)

#for k in range(features.shape[1]):
    #plt.scatter(k*np.ones(features[::20, 0].size), features[::20, k])
    #plt.show()

plt.show()

d_train = PyraatDataset2(actions[:-300, :], features[:-300, :])
d_val = PyraatDataset2(actions[-300:, :], features[-300:, :])
#print d_train.num_batches(50)


LOAD = True
TRAIN = True
EPOCHS = 90
save_dir = './trained/artnet'
test_name = 'acoustic_artnet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


input_dim, output_dim = d_train.parameter_sizes()
latent_size = 10
#print input_size, output_size, state_size 

#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#model = VAE( input_dim, latent_size, state_dim, output_dim, log_dir=log_dir, inner_width=20, auto_logging=False, sess=session)
model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=50, auto_logging=False, sess=session, lr=1e-3)

session.run(tf.global_variables_initializer())

#print('Training iteration #{0}'.format(n))


###
if LOAD:
    model.load(load_path)

if TRAIN:
    model.train(d_train, epochs=EPOCHS, batch_size=50, d_val=d_val)
    model.save(save_path)


y,x = d_val.get_batch(0, 50)

h = model.encode(y)

_x = model.decode(h)

plt.scatter(np.arange(50), np.mean(np.abs(x-_x)**2, axis=1))

#plt.scatter(y, x) 

#plt.scatter(y, _x, c='r')
plt.show()
