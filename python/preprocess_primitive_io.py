
import os
import numpy as np
import pylab as plt
import Artword as aw
from primitive.DataHandler import DataHandler
import scikits.talkbox.features as tb
from scikits.talkbox import segment_axis

from genfigures.plot_functions import *

#tikz_dir = '/home/jacob/Projects/Dissertation/Doc/tikz/'

#directory = "data/batch_zeros_100_10"
#directory = "data/batch_random_12_12"
directory = "data/rand_prim_1sec"

save_dir ="../tensorflow/primitive_io" 

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


dh = DataHandler()

nwin =240 # 30 ms
nstep=80 # 10 ms
overlap = nwin-nstep
for k in range(1, 1000):
    full_dir = directory+str(k)

    print "Loading utterance: ", k
    dh.LoadDataDir(directory=directory, min_index=k, max_index=k)

    print dh.raw_data.keys()
    if np.isnan(dh.raw_data['sound_wave']).any():
        print "Utterance contains NaNs"
        print "Skip!"
        print "#"*50
        continue 

    mfcc, mel_spectrum, spectrum = tb.mfcc(dh.raw_data['sound_wave'], nwin=nwin, nstep=nstep, nfft=512, nceps=13, fs=8000)
    
    print mfcc.shape

    print dh.raw_data['action_hist_1'].shape
    print dh.raw_data['state_hist_1'].shape

    #print dh.raw_data['art_hist'].shape
    #print mfcc.shape, mel_spectrum.shape, spectrum.shape

    action_segs = segment_axis(dh.raw_data['action_hist_1'][:, :], nwin/8, overlap/8, axis=1)
    action_segs = np.swapaxes(action_segs, 0, 1)

    state_segs = segment_axis(dh.raw_data['state_hist_1'][:, :], nwin/8, overlap/8, axis=1)
    state_segs = np.swapaxes(state_segs, 0, 1)

    print "Saving I/O data..."
    np.savez(save_dir+"/data"+str(k), 
             mfcc=mfcc, mel_spectrum = mel_spectrum, spectrum=spectrum, action_segs=action_segs, state_segs=state_segs)
                                                   

    #print "Saving animation..."
    #dh.SaveAnimation()
    #print "Done!"
    #print "#"*50

    
    
    #plt.figure()
    #plt.imshow(mfcc)
    #plt.figure()
    #plt.imshow(mel_spectrum)
    #plt.figure()
    #plt.imshow(spectrum)
    #plt.show()
    
