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


H = np.zeros((1, 20))
H_hat = np.zeros((1, 20))

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
        if filename[0] != 'b':
            continue
        data = np.load(directory+'/'+filename+'/model_data.npz')
        print data.keys()

        # load data into local variables
        y = data['y']
        x = data['x']
        h = data['h']
        y_hat = data['y_hat']
        x_hat = data['x_hat']
        h_hat = data['h_hat']
        h_std = data['h_std']

        H = np.append(H, h, axis=0)
        H_hat = np.append(H_hat, h_hat, axis=0)

        total_samples += y.shape[0]

        y_mean += np.sum(y, axis=0)
        y_error += np.sum(np.abs(y-y_hat), axis=0)
        x_error += np.sum(np.abs(x-x_hat), axis=0)
        h_error += np.sum(np.abs(h-h_hat), axis=0)

print total_samples
y_mean = y_mean/total_samples
#h_error = h_error[:10]

# Computing information capacity
cov_E = np.cov(H.T-H_hat.T)/4.
cov_H_hat = np.cov(H_hat.T)
cov_H= np.cov(H.T)

lambE, QE = np.linalg.eig(cov_E)
lambH, QH = np.linalg.eig(cov_H_hat)

AH  = np.dot(QE.T, np.dot(cov_H, QE))
AH_hat  = np.dot(QE.T, np.dot(cov_H_hat, QE))

a_H = np.zeros(20)
a_H_hat = np.zeros(20)

for k in range(20):
    a_H[k] = AH[k,k]
    a_H_hat[k] = AH_hat[k,k]



plt.figure()
plt.imshow(cov_E)
plt.title('$\\Sigma_E$')
tikz_save(directory+'/ICE_error_covariance.tikz',
            data_path='tikz/ICE/')

plt.figure()
plt.imshow(cov_H_hat)
plt.title('$\\Sigma_{\\hat{h}}$')
#plt.title('H_hat')
tikz_save(directory+'/ICE_output_covariance.tikz',
            data_path='tikz/ICE/')
#
#plt.figure()
#plt.imshow(cov_H)
#plt.title('H')
#plt.show()

#plt.figure()
#plt.plot(lambE)
#plt.plot(lambH)
#plt.plot(a_H-lambE)
#plt.title('H')
#plt.figure()
plt.figure()
plt.plot(lambE, 'r--', label='$\\lambda_E$')
#plt.plot(lambH)
plt.plot(a_H_hat, 'b', label='$A_{k,k}$')
plt.legend(loc='upper right')
plt.grid(True)

tikz_save(directory+'/ICE_waterfill.tikz',
            data_path='tikz/ICE/')
plt.show()

ind = a_H_hat>lambE
a_H_hat = a_H_hat[ind]
lambE = lambE[ind]
print np.log2(np.product(a_H_hat)), np.log2(np.product(lambE)), np.log2(np.product(a_H_hat))- np.log2(np.product(lambE))

exit()




plt.figure()
plt.bar(np.arange(h_std.size), h_std)
plt.ylabel('Decoder Variance')
plt.xlabel('Speech Primitive')
tikz_save(directory+'/primitive_std.tikz',
            data_path='tikz/ICE/')
plt.show()
plt.close()

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






