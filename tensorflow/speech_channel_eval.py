import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig
import scikits.talkbox.features as tb

import tensorflow as tf
from autoencoder.pyraat_vae import VAE, PyraatDataset, LoadData, variable_summaries

import PyRAAT as vt
#import primitive.RandomArtword as aw
import Artword as aw
from primitive.RandExcite import RandExcite

from genfigures.plot_functions import *
LOAD_ALL = False

def simulate(art_seq, directory):
    utterance_length = 1.0
    loops = 1

    free_muscles = np.array([aw.kArt_muscle.INTERARYTENOID ,
                    aw.kArt_muscle.MASSETER ,
                    aw.kArt_muscle.ORBICULARIS_ORIS ,
                    aw.kArt_muscle.MYLOHYOID ,
                    aw.kArt_muscle.STYLOGLOSSUS ,
                    aw.kArt_muscle.GENIOGLOSSUS])# ,
    free_muscles = np.sort(free_muscles)

    initial_art= np.zeros(aw.kArt_muscle.MAX)
    initial_art[aw.kArt_muscle.LUNGS] = 0.5
    initial_art[aw.kArt_muscle.LEVATOR_PALATINI] = 1.
    initial_art[free_muscles] = art_seq[0,:]

    # generate corresponding utterance
    utterance = RandExcite(directory=directory, 
                           loops=loops, 
                           utterance_length=utterance_length) #,
                           #initial_art=initial_art)

    utterance.InitializeAll(initial_art=initial_art,
                            random=False, addDTS=False)


    # force non free muscles to zero 
    time = np.arange(0.03, 1.01, 0.01)
    for k, muscle in enumerate(free_muscles):
        utterance.SetManualArticulation(muscle, time, art_seq[:,k])


    for k in range(aw.kArt_muscle.MAX):
        if k not in free_muscles:

            if k ==  aw.kArt_muscle.LUNGS:
                utterance.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 1.0], 
                                                                      [0.5, 0.0])

            elif k == aw.kArt_muscle.LEVATOR_PALATINI:
                utterance.SetManualArticulation(aw.kArt_muscle.LEVATOR_PALATINI,  [0.0, utterance_length], 
                                                                                  [1.0, 1.0])
            else:
                utterance.SetManualArticulation(k, [0.0, utterance_length], [0.0, 0.0])

    utterance.Run()

    mfcc_out, mel_spectrum, spectrum = tb.mfcc(utterance.GetSoundWave(), nwin=240, nstep=80, nfft=512, nceps=13, fs=8000)
    return mfcc_out

if LOAD_ALL:
    actions, features = LoadData(directory='speech_io', inputs='art_segs', outputs='mfcc', shuffle=False)

    #mean_f = np.mean(features, axis=0)
    min_f = np.min(features, axis=0)
    max_f = np.max(features, axis=0)
    features = (features-min_f)/(max_f - min_f)


    d_train = PyraatDataset(actions[:-100, :], features[:-100, :])
    d_val = PyraatDataset(actions[-100:, :], features[-100:, :])
    input_dim, output_dim = d_train.parameter_sizes()
    #print d_train.num_batches(50)

    np.savez('mfcc_norm', min_f=min_f, max_f=max_f, input_dim=input_dim, output_dim=output_dim)
else:
    d = np.load('mfcc_norm.npz')
    min_f = d['min_f']
    max_f = d['max_f']
    input_dim = d['input_dim']
    output_dim = d['output_dim']



save_dir = './trained/artnet'
test_name = 'artnet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


latent_size = 7
#print input_size, output_size, state_size 

# setup VAE 
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=50, auto_logging=False, sess=session, lr=1e-3)
session.run(tf.global_variables_initializer())
model.load(load_path)

#batch_ind = 5

rounds = 1000
mean_error = np.zeros(rounds)
coords = np.zeros((rounds, latent_size))
coords_out = np.zeros((rounds, 98, latent_size))
for k in range(rounds):
    h = np.zeros(latent_size)
    h[0] = np.random.normal()/2.
    h[1] = np.random.normal()/2.
    h[2] = np.random.normal()/2.
    #h = np.random.normal(size=7)/2.
    coords[k, :] = h
    h = np.tile(h, (98, 1))
    
    _x = model.decode(h)
    #_xp = model.decode(h+hs)
    #_xn = model.decode(h-hs)
    
    #error = np.abs(x-_x)
    #plt.plot(error)
    #plt.show()
    
    #for k in range(x.shape[1]):
    #    plt.figure()
    #    plt.plot(x[:, k], '-b')
    #    plt.plot(_x[:, k], '-r')
    #    plt.plot(_xp[:, k], '--r')
    #    plt.plot(_xn[:, k], '--r')
        
    
    directory = "speech_out/junk"
    
    # compute features from output sound
    mfcc_out = simulate(_x, directory)
    mfcc_out = (mfcc_out-min_f)/(max_f - min_f)
    
    #plt.figure()
    #plt.imshow(mfcc_out)
    ##plt.figure()
    ##plt.imshow(y_in)
    #
    #plt.show()
    
    _h = model.encode(mfcc_out)
    coords_out[k, :, :] = _h
    
    #for k in range(h.shape[1]):
    #    plt.figure()
    #    plt.plot(h[:, k], 'b')
    #    plt.plot(_h[:, k], 'r')

    mean_error[k] = np.mean(np.abs(h-_h)**2)
    
    #plt.show()

np.savez('3D_results', mean_error=mean_error, coords=coords, coords_out=coords_out)

for d in range(latent_size):
    plt.scatter(coords[:, d], mean_error)
    plt.show()
    
    

    #plt.scatter(y, x) 
    
    #plt.scatter(y, _x, c='r')
    #plt.show()
