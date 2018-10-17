
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

#import tensorflow as tf
#from autoencoder.pendulum_vae import VAE, Dynamical_Dataset, variable_summaries

from genfigures.plot_functions import *

  
class PyraatDataset:
    def __init__(self, directory='speech_io', outputs='mfcc', inputs='art_segs', train=True):

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
                actions = np.append(actions, data[inputs][:, :, -1], axis=0)
                #spec = np.append(spec, data['mel_spectrum'], axis=0)
            else:
                features = data[outputs]
                actions = data[inputs][:, :, -1]
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

        print np.sum(features[:, 0]>E_m), np.sum(features[:, 0]>E_m+E_sd)

        MultiPlotTraces(actions.T, np.arange(actions.shape[1]), actions.shape[0], 80)
        plt.show()

        self.features = features
        self.actions = actions
        
        #self.normalize()


    #def normalize(self):
    #    self._input_min = np.min(self._input)
    #    self._input_range = np.max(self._input)-self._input_min
    #    self._input = (self._input - self._input_min)/self._input_range

    #    self._output_min = np.min(self._output)
    #    self._output_range = np.max(self._output)-self._output_min
    #    self._output = (self._output - self._output_min)/self._output_range

    #def unnormalize(self, input_vals, output_vals):
    #    input_vals = input_vals*self._input_range+self._input_min
    #    output_vals = output_vals*self._output_range+self._output_min
    #    return input_vals, output_vals

    #def renormalize(self,input_vals, output_vals):
    #    input_vals = (input_vals-self._input_min)/self._input_range
    #    output_vals = (output_vals-self._output_min)/self._output_range
    #    return input_vals, output_vals

    def num_batches(self, batch_size):
        return int(np.floor(self.features.shape[0]/batch_size))

    #def num_points(self):
        #return self.mnist.num_examples

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
        return self.features[start:stop, :], self.features[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]

d = PyraatDataset(directory='speech_io', inputs='art_segs', outputs='mfcc')
print d.num_batches(50)
