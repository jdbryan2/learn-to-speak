

import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.pendulum_vae import VAE, variable_summaries
from double_pendulum import Pendulum

class OperatorDataset:
    def __init__(self, input_vals, operator, train=True):
        self._input = input_vals
        self._operator = operator
        self._output = operator(input_vals)

        self.normalize()


    def normalize(self):
        self._input_min = np.min(self._input)
        self._input_range = np.max(self._input)-self._input_min
        self._input = (self._input - self._input_min)/self._input_range

        self._output_min = np.min(self._output)
        self._output_range = np.max(self._output)-self._output_min
        self._output = (self._output - self._output_min)/self._output_range

    def unnormalize(self, input_vals, output_vals):
        input_vals = input_vals*self._input_range+self._input_min
        output_vals = output_vals*self._output_range+self._output_min
        return input_vals, output_vals

    def renormalize(self,input_vals, output_vals):
        input_vals = (input_vals-self._input_min)/self._input_range
        output_vals = (output_vals-self._output_min)/self._output_range
        return input_vals, output_vals

    def num_batches(self, batch_size):
        return int(np.floor(self._input.shape[0]/batch_size))

    #def num_points(self):
        #return self.mnist.num_examples

    def parameter_sizes(self):
        return self._input.shape[1], self._output.shape[1]

    def get_batch(self, n=0, batch_size=50):
        start = n*batch_size
        stop = (n+1)*batch_size
        #return self._input[start:stop, :], self._output[start:stop, :]
        return self._output[start:stop, :], self._input[start:stop, :]

    def data(self):
        start = 0
        stop = self._input.shape[0]
        return self._output[start:stop, :], self._input[start:stop, :]
        #return self._input[start:stop, :], self._output[start:stop, :]


LOAD = False
TRAIN = True
EPOCHS = 100
save_dir = './operator'
test_name = 'operator'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def cubic(_x):
    x = 2.*_x - 2
    return x**3 + x**2 - x - 1


d_train = OperatorDataset(np.random.random((1000, 1)), cubic)
d_val = OperatorDataset(np.random.random((100,1)), cubic)

#x, y = d_train.data()
#plt.scatter(x,y)
#x, y = d_val.data()
#plt.scatter(x,y, c='r')
#plt.show()

input_dim = 1 
output_dim = 1
latent_size = 20
#print input_size, output_size, state_size 

#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#model = VAE( input_dim, latent_size, state_dim, output_dim, log_dir=log_dir, inner_width=20, auto_logging=False, sess=session)
model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=20, auto_logging=False, sess=session,
            lr=1e-2)

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

plt.scatter(y, x) 

plt.scatter(y, _x, c='r')
plt.show()
