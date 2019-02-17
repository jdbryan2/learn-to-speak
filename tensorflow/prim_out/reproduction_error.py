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


directory = 'primtest3_1'

y_mean = 0
y_error = 0
x_error = 0
h_error = 0
total_samples = 0

d = np.load('norms.npz')
min_x = d['min_in']
max_x = d['max_in']
x_range = np.abs(max_x - min_x)
min_y = d['min_out']
max_y = d['max_out']
y_range = np.abs(max_y-min_y)
print min_y

if os.path.exists(directory):
    for filename in os.listdir(directory):
        print filename
        data = np.load(directory+'/'+filename+'/model_data.npz')
        print data.keys()

        # load data into local variables
        y = data['y']
        x = data['x']
        h = data['h']
        y_hat = data['y_hat']
        x_hat = data['x_hat']
        h_hat = data['h_hat']

        total_samples += y.shape[0]

        y_mean += np.sum(y, axis=0)
        y_error += np.sum(np.abs(y-y_hat), axis=0)
        x_error += np.sum(np.abs(x-x_hat), axis=0)
        h_error += np.sum(np.abs(h-h_hat), axis=0)

print total_samples
y_mean = y_mean/total_samples
h_error = h_error[:10]

plt.figure()
plt.bar(np.arange(h_error.size), 100*h_error/total_samples)
plt.ylabel('\\% Error')
plt.xlabel('Speech Primitive')
tikz_save(directory+'/primitive_error.tikz',
            data_path='tikz/ICE/')
plt.close()

plt.figure()
plt.bar(np.arange(x_error.size),100*x_error/total_samples/x_range)
plt.ylim([0, 25])
plt.ylabel('\\% Error')
plt.xlabel('Articulatory Primitive Action')
tikz_save(directory+'/control_error.tikz',
            data_path='tikz/ICE/')
plt.close()

plt.figure()
plt.bar(np.arange(y_error.size),100*y_error/total_samples/y_range)
plt.ylim([0, 25])
plt.ylabel('\\% Error')
plt.xlabel('MFCC')
tikz_save(directory+'/mfcc_error.tikz',
            data_path='tikz/ICE/')

plt.close()
#plt.show()






