
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.primitive_vae import VAE, PyraatDataset2, LoadData, variable_summaries

from genfigures.plot_functions import *

def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    data = (data-min_val)/(max_val - min_val)
    return data, min_val, max_val

def denormalize(data, min_val, max_val):
    return data*(max_val - min_val)+min_val

inputs, outputs, states = LoadData(directory='primitive_io', inputs_name='action_segs', states_name='state_segs',
                                   outputs_name='mfcc')

inputs = np.append(inputs, states, axis=1)

#mean_f = np.mean(features, axis=0)
#min_f = np.min(outputs, axis=0)
#max_f = np.max(outputs, axis=0)
#outputs = (outputs-min_f)/(max_f - min_f)
outputs, min_out, max_out = normalize(outputs)
inputs, min_in, max_in = normalize(inputs)
#states, min_state, max_state = normalize(states)


#for k in range(features.shape[1]):
    #plt.scatter(k*np.ones(features[::20, 0].size), features[::20, k])
    #plt.show()
plt.figure()
plt.plot(inputs[:1000, :])
plt.figure()
plt.plot(outputs[:1000, :])
#plt.figure()
#plt.plot(states[:1000, :])

plt.show()

val_ratio = 0.1
total = inputs.shape[0]
val_count = int(np.floor(total*val_ratio))
print val_count

d_train = PyraatDataset2(inputs[:-val_count, :], outputs[:-val_count, :])
d_val = PyraatDataset2(inputs[-val_count:, :], outputs[-val_count:, :])
#print d_train.num_batches(50)

#exit()
LOAD = True
TRAIN = True
EPOCHS = 10
save_dir = './trained/primnet'
test_name = 'primnet'
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
