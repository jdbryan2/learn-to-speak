
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

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--stop', type=int, default=100)

args = parser.parse_args()
print "start: ", args.start
print "stop: ", args.stop

#exit()
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

    sound = prim.GetSoundWave()
    total_energy = np.sum((sound[1:] - sound[:-1])**2)

    return mfcc_out, total_energy


test_name = 'primtest_1'
#test_name = 'primtest1_5'
#test_name = 'primtest2_1'
#test_name = 'primtest2_5'
test_name = 'primtest3_1'
#test_name = 'primtest3_5'
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

LOAD_DATA = False

# compute normalization parameters from data if needed
if os.path.exists(save_dir+'/norms.npz') and LOAD_DATA==False:

    d = np.load(save_dir+'/norms.npz')
    min_x = d['min_in']
    max_x = d['max_in']
    min_y = d['min_out']
    max_y = d['max_out']
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
#batch_ind = args.batch

#for batch_ind in range(args.start, args.stop):
batch_ind = args.start
while batch_ind < args.stop:
    print "#"*80
    print "Batch %i/%i"%(batch_ind, args.stop-args.start)
    print "#"*80
    directory = "prim_out/"+test_name+"/symbols/s"+str(batch_ind) # adjust so it matches with original data

    save_data = {}

    # load directly from original data file
    #######################################
    #data = np.load("../python/data/rand_prim_1sec/data%i.npz"%batch_ind)
    #y_in, mel_spectrum, spectrum = tb.mfcc(data['sound_wave'], nwin=240, nstep=80, nfft=512, nceps=13, fs=8000)
    #y_in, _, _ = normalize(y_in, min_val=min_out, max_val=max_out) # confusing notation on in vs out -> min_out == output of vocal tract

    # compute average variance on each primitive dimension
    tx = np.random.normal(0, 2, (1, latent_size))
    tx = np.tile(tx, (98, 1))

    delta = 20
    for k in range(tx.shape[1]):
        tx[:delta, k] = np.linspace(0, tx[-1, k], delta)
        #plt.plot(tx[:, k])
        #plt.show()

    #h_std = model.encode_std(tx)
    #h_std = np.exp(h_std)
    #save_data['h_std'] = h_std # save standard deviation

    x = model.decode(tx)
    x = denormalize(x, min_val=min_x, max_val=max_x)

    y, energy = simulate(x[:, :10], directory)
    y = tile_gestures(y, gesture_length)
    y, _, _ = normalize(y, min_val=min_y[:y.shape[1]], max_val=max_y[:y.shape[1]])

    rx = model.encode(y)
    rx_std = model.encode_std(y)
    rx_std = np.exp(rx_std)
    #h_hat = moving_average(h_hat, n=10)


    #error = np.abs(tx-rx)
    #mean_error = np.cumsum(error, axis=0)
    #for t in range(error.shape[0]):
    #    mean_error[t, :] = error[t, :]/(t+1)
    #
    #for k in range(rx.shape[1]):
    ##for k in range(10):
    ##    plt.plot(mean_error[:, k])
    #    plt.plot(tx[:, k])
    #    plt.plot(rx[:, k], 'r')
    #    plt.plot(rx[:, k]-rx_std[:, k], 'r--')
    #    plt.plot(rx[:, k]+rx_std[:, k], 'r--')
    #plt.show()
    if energy > 10**-3:
        batch_ind += 1
        np.savez(directory+'/model_data', tx=tx, rx=rx, rx_std=rx_std)
    else:
        print "Not enough pylons"

