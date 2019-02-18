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
X = np.zeros((1, 20))
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
        X = np.append(X, np.mean(tx[50:, :], axis=0).reshape((1,20)), axis=0)
        cum_error = np.cumsum(error[20:, :], axis=0)
        for t in range(cum_error.shape[0]):
            cum_error[t, :] = cum_error[t, :]/(t+1)

        #for k in range(cum_error.shape[1]):
        #    plt.plot(cum_error[:, k], 'b')
        #    plt.plot(tx[:, k], 'b--')
        #    plt.plot(rx[:, k], 'r--')
        #    plt.show()

        mean_error += cum_error
        #mean_error += np.abs(error)#**2

E = E[1:, :] # remove first row of zeros
X = X[1:, :] # remove first row of zeros
Y = Y[1:, :] # remove first row of zeros

cov_E  = np.cov(E.T)
cov_Y  = np.cov(Y.T)
cov_X = np.cov(X.T)
dE = np.linalg.det(cov_E)
dY = np.linalg.det(cov_Y)
I = np.log2(dY/dE)
print I
print np.log2(dY)
print np.log2(dE)

std = std/total_samples
#plt.plot(std)
#plt.show()
#plt.imshow(np.abs(cov_E))
#plt.show()
#plt.imshow(np.abs(cov_Y))
#plt.show()
cov_E /= 4.
cov_X /= 4.
lamb, Q = np.linalg.eig(cov_E)
lamby, Qy = np.linalg.eig(cov_Y)
#lamb = lamb-0.4
print lamb
print lamby
A  = np.dot(Q.T, np.dot(cov_Y, Q))
plt.imshow(np.abs(A))
plt.show()

#print np.log2(np.linalg.det(cov_X+cov_E)/np.linalg.det(cov_E))
le = lamb[lamby>lamb]
ly = lamby[lamby>lamb]
print np.log2(np.product(ly)/np.product(le)), np.log2(np.product(le)), np.log2(np.product(ly))

A = np.dot(Q.T, np.dot(cov_X, Q))
v = np.zeros(A.shape[0])
a = np.zeros(A.shape[0])
x = np.zeros(A.shape[0])
e = np.zeros(A.shape[0])
for k in range(A.shape[0]):
    #print cov_X[k,k]
    if A[k, k]> 0:
        v[k] = A[k,k] + lamb[k]
        a[k] = A[k,k]
        e[k] = lamb[k]
        x[k] = cov_X[k,k]

v = v[v>0]
e = e[e>0]
print 'v', np.product(v)
print 'e', np.product(e)

#plt.plot(v)
plt.plot(a)
plt.plot(lamb)
plt.plot(lamby)
plt.show()

print cov_E.shape
mean_error = mean_error/total_samples

mean_error = np.delete(mean_error, [8,9], 1)

ind = np.argsort(abs(mean_error[-1, :]))

rows = np.arange(mean_error.shape[1])
# # generate error plot for dissertation
#plt.figure()
#PlotTraces(np.abs(mean_error).T, rows, max_length=mean_error.shape[0], sample_period=80, highlight=0)
##plt.plot([0.2, 0.2], [0, 1], 'r--')
#plt.xlabel("Time (s)")
#plt.ylabel("Mean Square Error")
#plt.ylim([0, 1])
#tikz_save(directory+'/timing_error.tikz',
#            data_path='tikz/ICE/')
#plt.close()
#plt.show()

#print mean_error[-1, ind]
#print np.log2(1./np.abs(mean_error[-1, ind]))
#print np.cumsum(np.log2(1./np.abs(mean_error[-1, ind])))

#
#for k in rows:
#    print k
#    plt.figure()
#    PlotTraces(np.abs(mean_error).T, rows, max_length=mean_error.shape[0], sample_period=80, highlight=k)
#    plt.title(k)
#plt.show()

#for k in range(mean_error.shape[1]):
#for k, dex in enumerate(ind):
#    #if std[k] < 0.06:
#    if k>10:
#        plt.plot(np.abs(mean_error[:, dex]), '--b', alpha=0.3)
#    else:
#        plt.plot(np.abs(mean_error[:, dex]), 'b')
#
#plt.show()
