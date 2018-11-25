import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D


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

def mnist_examples(labels, img_in, img_out, select=0, count=10):
    ind, = np.where(labels==select)

    #plt.figure()
    img_handle = mnist_sequence(img_in[ind[1:count]], img_out[ind[1:count]])
    img_handle.set_cmap('Greys')
    plt.axis('off')

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

def channel_capacity(tx, rx):

    power = np.mean(np.sum(np.abs(tx)**2, axis=1))
    MSE = np.mean(np.sum(np.abs(tx-rx)**2, axis=1))
    chan_cap = np.sum(0.5*np.log2(1+tx.shape[1]/MSE))
    return chan_cap, MSE, power
