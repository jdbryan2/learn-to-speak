
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.primitive_vae import VAE, variable_summaries
from data_loader import PyraatDataset, LoadData

from genfigures.plot_functions import *
from helper_functions import *

# VAE parameters
LOAD = True
TRAIN = True
EPOCHS = 10
SAVE_INTERVAL = 1

latent_size = 10
inner_width = 100
gesture_length = 1
#beta = 1.*latent_size/inner_width
beta = 0.001

test_name = "artgest_%i"%gesture_length
#test_name = "primgest_%i"%gesture_length

save_dir = './trained/' + test_name
log_dir = None #save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'
# Create save_dir if it does not already exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    LOAD = False # don't load if we just had to create the save directory


if test_name[:3] == 'art':
    # states get ignored so just point them at something
    inputs, outputs, states = LoadData(directory='speech_io',
                                       inputs_name='art_segs',
                                       outputs_name='mfcc',
                                       states_name='art_segs',
                                       shuffle=True,
                                       seq_length=gesture_length)

elif test_name[:4] == 'prim':

    inputs, outputs, states = LoadData(directory='primitive_io',
                                       inputs_name='action_segs',
                                       states_name='state_segs',
                                       outputs_name='mfcc',
                                       shuffle=True,
                                       seq_length=gesture_length)

    # append state variables to inputs
    inputs = np.append(inputs, states, axis=1)

else:
    print "You done fucked up."
    exit()


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
             log_dir=log_dir, inner_width=inner_width, beta=beta,
             auto_logging=False, sess=session, lr=1e-3)

session.run(tf.global_variables_initializer())


###
# TODO: Add printout to show how dimensions are being used on each iter
###
if LOAD:
    print "Loading previous model."
    model.load(load_path)

if TRAIN:
    total_epochs = 0
    while total_epochs < EPOCHS:
        print "Total Epochs Trained: " + str(total_epochs)
        if SAVE_INTERVAL > EPOCHS-total_epochs:
            model.train(d_train, epochs=EPOCHS-total_epochs, batch_size=50, d_val=d_val)
            total_epochs += EPOCHS-total_epochs
        else:
            model.train(d_train, epochs=SAVE_INTERVAL, batch_size=50, d_val=d_val)
            total_epochs += SAVE_INTERVAL
        model.save(save_path)

        y,x = d_train.get_batch(0, 50)

        h_std = model.encode_std(y)

        unused_dims = np.exp(np.mean(h_std, axis=0))>0.9
        print("Unused dimensions: %i/%i"%(np.sum(unused_dims), latent_size))
        print(np.exp(np.mean(h_std, axis=0)))

    #plt.figure()
    #plt.plot(np.exp(np.mean(h_std, axis=0)))


# End Training
####################

# print out unused dimensions

# identify unused dimensions
y,x = d_train.get_batch(0, 50)

h_std = model.encode_std(y)

unused_dims = np.exp(np.mean(h_std, axis=0))>0.9
print("Unused dimensions: %i/%i"%(np.sum(unused_dims), latent_size))

plt.figure()
plt.plot(np.exp(np.mean(h_std, axis=0)))



print('Saving model params')
if LOAD==True and os.path.exists(save_dir+'/params.npz'):
    params = np.load(save_dir+'/params.npz')
    print 'Total EPOCHS: ', EPOCHS+params['EPOCHS']
    np.savez(save_dir+'/params',
             EPOCHS=EPOCHS+params['EPOCHS'], # tally up epochs
             latent_size=latent_size,
             inner_width=inner_width,
             input_dim=input_dim,
             output_dim=output_dim,
             val_ratio=val_ratio,
             gesture_length=gesture_length,
             beta=beta)
else:
    # overwrite or simply write new file
    print 'Total EPOCHS: ', EPOCHS
    np.savez(save_dir+'/params',
             EPOCHS=EPOCHS,
             latent_size=latent_size,
             inner_width=inner_width,
             input_dim=input_dim,
             output_dim=output_dim,
             val_ratio=val_ratio,
             gesture_length=gesture_length,
             beta=beta)



#plt.scatter(np.arange(50), np.mean(np.abs(x-_x)**2, axis=1))
plt.show()
