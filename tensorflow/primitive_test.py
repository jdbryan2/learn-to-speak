
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.primitive_vae import VAE, PyraatDataset2, LoadData, variable_summaries

from genfigures.plot_functions import *
from helper_functions import *

# VAE parameters
LOAD = True
TRAIN = True
EPOCHS = 10
save_dir = './trained/primnet'
test_name = 'primnet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'

latent_size = 20
inner_width = 50

# Create save_dir if it does not already exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

    np.savez(save_dir+'/params', )


# load data
inputs, outputs, states = LoadData(directory='primitive_io', inputs_name='action_segs', states_name='state_segs',
                                   outputs_name='mfcc', shuffle=True)

# append state variables to inputs
inputs = np.append(inputs, states, axis=1)

# normalize to range of [0, 1]
outputs, min_out, max_out = normalize(outputs)
inputs, min_in, max_in = normalize(inputs)


val_ratio = 0.1 # validation ratio
total = inputs.shape[0]
val_count = int(np.floor(total*val_ratio)) # validation count
print val_count

# create datasets (PyraatDataset2 appends inputs and outputs together internally)
d_train = PyraatDataset(inputs[:-val_count, :], outputs[:-val_count, :])
d_val = PyraatDataset(inputs[-val_count:, :], outputs[-val_count:, :])

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

input_dim, output_dim = d_train.parameter_sizes()

####################
# Train Neural Net
####################

session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, output_dim,
             log_dir=log_dir, inner_width=inner_width, auto_logging=False, sess=session, lr=1e-3)

session.run(tf.global_variables_initializer())


###
if LOAD:
    model.load(load_path)

if TRAIN:
    model.train(d_train, epochs=EPOCHS, batch_size=50, d_val=d_val)
    model.save(save_path)


# End Training
####################

# print out unused dimensions

# identify unused dimensions
y,x = d_val.get_batch(0, 50)

h_std = model.encode_std(y)

unused_dims = np.exp(np.mean(h_std, axis=0))>0.9
print("Unused dimensions:")
print(unused_dims)
plt.figure()
plt.plot(np.exp(np.mean(h_std, axis=0)))



print('Saving model params')
if LOAD==True and os.path.exists(save_dir+'/params.npz'):
    params = np.load(save_dir+'params.npz')
    np.savez(save_dir+'/params',
             EPOCHS=EPOCHS+params['EPOCHS'], # tally up epochs
             latent_size=latent_size,
             inner_width=inner_width,
             input_dim=input_dim,
             output_dim=output_dim,
             val_ratio=val_ratio)
else:
    # overwrite or simply write new file
    np.savez(save_dir+'/params',
             EPOCHS=EPOCHS, 
             latent_size=latent_size,
             inner_width=inner_width,
             input_dim=input_dim,
             output_dim=output_dim)


#plt.scatter(np.arange(50), np.mean(np.abs(x-_x)**2, axis=1))
#plt.show()
