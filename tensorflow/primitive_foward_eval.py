
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
from autoencoder.primitive_vae import VAE, PyraatDataset2, LoadData, variable_summaries

from genfigures.plot_functions import *




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
        for t in range(act_seq.shape[0]):
            prim._act.SetManualTarget(k, act_seq[t,k], time[t]) # muscle, target, time

    prim.InitializeControl(initial_art = prim.GetControl(prim._act.GetAction(time=0.0)))

    prim.Simulate()
    prim.SaveOutputs()
    mfcc_out, mel_spectrum, spectrum = tb.mfcc(prim.GetSoundWave(), nwin=240, nstep=80, nfft=512, nceps=13, fs=8000)
    return mfcc_out

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

def denormalize(data, min_val, max_val):
    return data*(max_val - min_val)+min_val




LOAD = True
EPOCHS = 10
save_dir = './trained/primnet'
test_name = 'primnet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'
latent_size = 10

LOAD_ALL = True

if LOAD_ALL:
    inputs, outputs, states = LoadData(directory='primitive_io', inputs_name='action_segs', states_name='state_segs',
                                       outputs_name='mfcc', shuffle=False)

    inputs = np.append(inputs, states, axis=1)

    outputs, min_out, max_out = normalize(outputs)
    inputs, min_in, max_in = normalize(inputs)

    d_train = PyraatDataset2(inputs[:-100, :], outputs[:-100, :])
    input_dim, output_dim = d_train.parameter_sizes()

    np.savez('prim_norm', min_out=min_out, 
                          max_out=max_out, 
                          min_in=min_in, 
                          max_in=max_in, 
                          input_dim=input_dim, 
                          output_dim=output_dim)
else:
    d = np.load('prim_norm.npz')
    min_in = d['min_in']
    max_in = d['max_in']
    min_out = d['min_out']
    max_out = d['max_out']
    input_dim = d['input_dim']
    output_dim = d['output_dim']


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# load up VAE
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=50, auto_logging=False, sess=session, lr=1e-3)

session.run(tf.global_variables_initializer())

model.load(load_path)

batch_ind = 0
directory = "prim_out/b"+str(batch_ind)

#data = np.load("../python/data/rand_prim_1sec/data%i.npz"%batch_ind)
#y_in, mel_spectrum, spectrum = tb.mfcc(data['sound_wave'], nwin=240, nstep=80, nfft=512, nceps=13, fs=8000)
#y_in, _, _ = normalize(y_in, min_val=min_out, max_val=max_out) # confusing notation on in vs out -> min_out == output of vocal tract 
y,x = d_train.get_batch(batch_ind, 98)
x = x[:, :20]

x_in = denormalize(x, min_val=min_in, max_val=max_in)
y_out = denormalize(y, min_val=min_out, max_val=max_out)

y_sim = simulate(x_in[:, :10], directory)

plt.figure()
plt.imshow(y_sim)
plt.figure()
plt.imshow(y_out)
plt.show()


h = model.encode(y)
hs = model.encode_std(y)

_x = model.decode(h)
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
mfcc_out, _, _ = normalize(mfcc_out, min_val=min_out, max_val=max_out) # confusing notation on in vs out -> min_out == output of vocal tract

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
