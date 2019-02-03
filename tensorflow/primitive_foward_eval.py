
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig
import scikits.talkbox.features as tb

from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance
from primitive.ActionSequence import ActionSequence

import tensorflow as tf
from autoencoder.primitive_vae import VAE, variable_summaries
from data_loader import PyraatDataset, LoadData, tile_gestures

from genfigures.plot_functions import *
from helper_functions import *


# function for quickly simulating and returning corresponding MFCC sequence


def simulate(act_seq, directory):
    utterance_length = 1.0
    loops = 1

    prim_dirname = '../python/data/batch_random_20_5'
    ind= get_last_index(prim_dirname, 'round')
    prim_filename = 'round%i.npz'%ind

    prim = PrimitiveUtterance()
    prim.LoadPrimitives(fname=prim_filename, directory = prim_dirname)

    prim.utterance = Utterance(directory = directory,
                               utterance_length=utterance_length,
                               loops=loops,
                               addDTS=False)

    prim._act = ActionSequence(dim=prim._dim,
                               initial_action=act_seq[0, :], #initial_action=np.zeros(prim._dim),
                               random=False)

    # load targets from act_seq
    time = np.arange(0.03, 1.01, 0.01)
    for k in range(act_seq.shape[1]):
        for t in range(min(act_seq.shape[0], time.size)):
            prim._act.SetManualTarget(k, act_seq[t,k], time[t]) # muscle, target, time

    prim.InitializeControl(initial_art = prim.GetControl(prim._act.GetAction(time=0.0)))

    prim.Simulate()
    prim.SaveOutputs()
    mfcc_out, mel_spectrum, spectrum = tb.mfcc(prim.GetSoundWave(), nwin=240, nstep=80, nfft=512, nceps=13, fs=8000)
    return mfcc_out


#test_name = 'primtest2_5'
#test_name = 'primtest3_1'
test_name = 'primtest3_5'
save_dir = './trained/'+test_name
load_path = save_dir+'/'+test_name+'.ckpt'
log_dir = save_dir+'/'+test_name+'_logs' # needed parameter for VAE model

print "Loading network paramters."
params = np.load(save_dir+'/params.npz')
EPOCHS = params['EPOCHS']
latent_size = params['latent_size']
inner_width = params['inner_width']
input_dim = params['input_dim']
output_dim = params['output_dim']
gesture_length = params['gesture_length']
beta = params['beta']

LOAD_DATA = True
# compute normalization parameters from data if needed
if os.path.exists(save_dir+'/norms.npz') and LOAD_DATA==False:

    d = np.load(save_dir+'/norms.npz')
    min_in = d['min_in']
    max_in = d['max_in']
    min_out = d['min_out']
    max_out = d['max_out']
    input_dim = d['input_dim']
    output_dim = d['output_dim']

else:

    inputs, outputs, states = LoadData(directory='primitive_io',
                                       inputs_name='action_segs',
                                       states_name='state_segs',
                                       outputs_name='mfcc', shuffle=False,
                                       seq_length=gesture_length)

    # append states to input if the loaded input dim is larger than the primitive inputs alone
    # "output_dim" has to do with VAE output, not channel output (vae output is channel input...)
    if output_dim > inputs.shape[1]:
        #print "appending states" 
        #print input_dim, output_dim,inputs.shape[1], outputs.shape[1]
        inputs = np.append(inputs, states, axis=1)
        #exit()

    outputs, min_out, max_out = normalize(outputs)
    inputs, min_in, max_in = normalize(inputs)

    d_train = PyraatDataset(inputs[:-100, :], outputs[:-100, :])
    input_dim, output_dim = d_train.parameter_sizes()

    np.savez(save_dir+'/norms', min_out=min_out,
                                max_out=max_out,
                                min_in=min_in,
                                max_in=max_in,
                                input_dim=input_dim,
                                output_dim=output_dim)


# load up VAE
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=inner_width, auto_logging=False, sess=session, lr=1e-3, beta=beta)

session.run(tf.global_variables_initializer())

model.load(load_path)

# load specific batch from data (make sure data is not randomized when loaded)
batch_ind = 2
directory = "prim_out/b"+str(batch_ind)

# load directly from original data file
#######################################
#data = np.load("../python/data/rand_prim_1sec/data%i.npz"%batch_ind)
#y_in, mel_spectrum, spectrum = tb.mfcc(data['sound_wave'], nwin=240, nstep=80, nfft=512, nceps=13, fs=8000)
#y_in, _, _ = normalize(y_in, min_val=min_out, max_val=max_out) # confusing notation on in vs out -> min_out == output of vocal tract

# get specific batch from dataset
y,x = d_train.get_batch(batch_ind, 98)
#x = x[:, :20] # this was product of crappy dataset design


h = model.encode(y)
_x = model.decode(h)
x_in = denormalize(_x, min_val=min_in, max_val=max_in)

y_out = denormalize(y, min_val=min_out, max_val=max_out)

if gesture_length>1:
    last_row = x_in[-1, :]
    x_in = x_in[::gesture_length, :]
    x_in = x_in.reshape((-1, x_in.shape[1]/gesture_length))
    #x_in = np.append(x_in, last_row[:x_in.shape[1]].reshape((1, -1)), axis=0)

y_sim = simulate(x_in[:, :10], directory)


plt.figure()
plt.imshow(y_sim)
plt.figure()
plt.imshow(y_out)
plt.show()


#h = model.encode(y)
hs = model.encode_std(y)

min_h = h-np.exp(hs)
max_h = h+np.exp(hs)

#for k in range(h.shape[1]):
#    plt.plot(h[:, k])
#    plt.plot(min_h[:, k], '--r')
#    plt.plot(max_h[:, k], '--r')
#    plt.show()


y_sim = tile_gestures(y_sim, gesture_length)
plt.imshow(y_sim)
plt.show()
y_sim, _, _ = normalize(y_sim, min_val=min_out[:y_sim.shape[1]], max_val=max_out[:y_sim.shape[1]])


h_hat = model.encode(y_sim)
hs = model.encode_std(y_sim)

min_h = h_hat-np.exp(hs)
max_h = h_hat+np.exp(hs)

for k in range(h.shape[1]):
    plt.plot(h[:, k])
    plt.plot(h_hat[:, k], 'r')
    plt.plot(min_h[:, k], '--r')
    plt.plot(max_h[:, k], '--r')
    plt.show()

#_x = model.decode(h)
#_xp = model.decode(h+hs)
#_xn = model.decode(h-hs)

#error = np.abs(x-_x)
#plt.plot(error)
#plt.show()

for k in range(10):
    plt.figure()
    plt.plot(x[:, k], '-b')
    plt.plot(_x[:, k], '-r')
#    plt.plot(_xp[:, k], '--r')
#    plt.plot(_xn[:, k], '--r')

plt.show()



exit()
directory = "prim_out/b"+str(batch_ind)

# compute features from output sound
mfcc_out = simulate(_x, directory)
mfcc_out = tile_gestures(mfcc_out, gesture_length)
mfcc_out, _, _ = normalize(mfcc_out, min_val=min_out[:mfcc_out.shape[1]], max_val=max_out[:mfcc_out.shape[1]]) # confusing notation on in vs out -> min_out == output of vocal tract

plt.figure()
plt.imshow(mfcc_out)
plt.figure()
plt.imshow(y_in)

plt.show()

_h = model.encode(mfcc_out)

for k in range(h.shape[1]):
    plt.plot(h[:, k], 'b')
    plt.plot(_h[:, k], 'r')
    plt.show()

#plt.scatter(y, x)

#plt.scatter(y, _x, c='r')
#plt.show()
