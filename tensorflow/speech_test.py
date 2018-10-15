
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

#import tensorflow as tf
#from autoencoder.pendulum_vae import VAE, Dynamical_Dataset, variable_summaries

from genfigures.plot_functions import *

ind_list = np.array(get_index_list(directory='speech_io', base_name='data'))

ind_list = np.sort(ind_list)

mfcc = np.array([])
spec = np.array([])
print "Loading Data..."
for k in ind_list:
    data = np.load("speech_io/data%i.npz"%k)

    if mfcc.size:
        mfcc = np.append(mfcc, data['mfcc'], axis=0)
        #spec = np.append(spec, data['mel_spectrum'], axis=0)
    else:
        mfcc = data['mfcc']
        #spec = data['mel_spectrum']

#plt.imshow(spec.T, aspect=3*20)
#plt.show()
mfcc = mfcc[~np.isnan(mfcc)].reshape((-1, 13))

E_m = np.mean(mfcc[:, 0])
E_sd = np.std(mfcc[:, 0])
print E_m, E_sd

print np.sum(mfcc[:, 0]>E_m), np.sum(mfcc[:, 0]>E_m+E_sd)

    
    
