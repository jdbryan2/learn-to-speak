import numpy as np
import os
import pylab as plt
import scipy.signal as signal
from genfigures.plot_functions import *

def normalize(data, **kwargs):
    if 'min_val' in kwargs:
        min_val = kwargs['min_val']
    else:
        min_val = np.min(data, axis=0)

    if 'max_val' in kwargs:
        max_val = kwargs['max_val']
    else:
        max_val = np.max(data, axis=0)

    data = (data-min_val)/(max_val - min_val)
    return data, min_val, max_val


directory = 'primtest3_1/symbols_bak2'
directory = 'primtest3_1/symbols'

total_samples = 0
mean_error = 0
E = np.zeros((1, 20))
Y = np.zeros((1, 20))
std = 0

if os.path.exists(directory):
    for filename in os.listdir(directory):
        print filename
        if not os.path.exists(directory+'/'+filename+'/model_data.npz'):
            continue
        data = np.load(directory+'/'+filename+'/model_data.npz')
        print data.keys()

        # load data into local variables
        tx = data['tx']
        rx = data['rx']
        rx_std = data['rx_std']
        total_samples += 1

        error = tx-rx
        std += np.mean(rx_std, axis=0)
        E = np.append(E, error[-1, :].reshape((1,20)), axis=0)
        Y = np.append(Y, np.mean(rx[50:, :], axis=0).reshape((1,20)), axis=0)
        cum_error = np.cumsum(error, axis=0)
        for t in range(error.shape[0]):
            cum_error[t, :] = cum_error[t, :]/(t+1)

        #for k in range(cum_error.shape[1]):
        #    plt.plot(cum_error[:, k], 'b')
        #    plt.plot(tx[:, k], 'b--')
        #    plt.plot(rx[:, k], 'r--')
        #    plt.show()

        mean_error += cum_error

E = E[1:, :] # remove first row of zeros
Y = Y[1:, :] # remove first row of zeros

cov_E  = np.cov(E.T)
cov_Y  = np.cov(Y.T)
dE = np.linalg.det(cov_E)
dY = np.linalg.det(cov_Y)
I = np.log2(dY/dE)
print I
print np.log2(dY)
print np.log2(dE)

std = std/total_samples
plt.plot(std)
plt.show()
pl.imshow(np.abs(cov_E))
plt.show()
plt.imshow(np.abs(cov_Y))
plt.show()
print cov_E.shape
mean_error = mean_error/total_samples

for k in range(mean_error.shape[1]):
    if std[k] < 0.06:
        plt.plot(np.abs(mean_error[:, k]))

plt.show()
