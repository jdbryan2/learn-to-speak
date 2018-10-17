
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.pendulum_vae import VAE, Dynamical_Dataset, variable_summaries

from genfigures.plot_functions import *

def LoadData(directory='speech_io', outputs='mfcc', inputs='art_segs', train=True):

    ind_list = np.array(get_index_list(directory=directory, base_name='data'))

    ind_list = np.sort(ind_list)

    features = np.array([])
    spec = np.array([])
    actions = np.array([])
    print "Loading Data..."
    for k in ind_list:
        data = np.load("speech_io/data%i.npz"%k)

        if features.size:
            features = np.append(features, data[outputs], axis=0)
            actions = np.append(actions, data[inputs][:, 1:, -1], axis=0)
            #spec = np.append(spec, data['mel_spectrum'], axis=0)
        else:
            features = data[outputs]
            actions = data[inputs][:, 1:, -1]
            #spec = data['mel_spectrum']

    #plt.imshow(spec.T, aspect=3*20)
    #plt.show()
    remove = ~np.isnan(features)
    features = features[remove].reshape((-1, 13))
    remove.reshape((-1, 13))
    actions = actions[remove[:, 0], :]
    #print remove.shape

    E_m = np.mean(features[:, 0])
    E_sd = np.std(features[:, 0])
    print E_m, E_sd
    m_count = np.sum(features[:, 0]>E_m)
    sd_count = np.sum(features[:, 0]>E_m+E_sd)
    count = features.shape[0]

    print 1.*m_count/count, 1.*sd_count/count

    MultiPlotTraces(actions.T, np.arange(actions.shape[1]), actions.shape[0], 80)
    plt.show()

    new_index = np.arange(0, actions.shape[0]).astype('int')
    np.random.shuffle(new_index)

    return actions[new_index, :], features[new_index, :]

  
class PyraatDataset:
    def __init__(self, actions, features):
        self.actions = actions
        self.features = features

    def num_batches(self, batch_size):
        return int(np.floor(self.features.shape[0]/batch_size))

    def parameter_sizes(self):
        return self.features.shape[1], self.actions.shape[1]

    def get_batch(self, n=0, batch_size=50):
        start = n*batch_size
        stop = (n+1)*batch_size
        return self.features[start:stop, :], self.actions[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]

    def data(self):
        start = 0
        stop = self.actions.shape[0]
        return self.features[start:stop, :], self.actions[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]


actions, features = LoadData(directory='speech_io', inputs='art_segs', outputs='mfcc')
d_train = PyraatDataset(actions[:-100, :], features[:-100, :])
d_val = PyraatDataset(actions[-100:, :], features[-100:, :])
print d_train.num_batches(50)


LOAD = False
TRAIN = True
EPOCHS = 1
save_dir = './trained/artnet'
test_name = 'artnet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


input_dim, output_dim = d_train.parameter_sizes()
latent_size = 3
#print input_size, output_size, state_size 

#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#model = VAE( input_dim, latent_size, state_dim, output_dim, log_dir=log_dir, inner_width=20, auto_logging=False, sess=session)
model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=50, auto_logging=False, sess=session, lr=1e-2)

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

#plt.scatter(y, x) 

#plt.scatter(y, _x, c='r')
#plt.show()
