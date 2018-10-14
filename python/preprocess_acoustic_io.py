
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
directory = "data/test/breathe_rand_init_"

free_muscles = np.array([aw.kArt_muscle.INTERARYTENOID ,
                aw.kArt_muscle.MASSETER ,
                aw.kArt_muscle.ORBICULARIS_ORIS ,
                aw.kArt_muscle.MYLOHYOID ,
                aw.kArt_muscle.STYLOGLOSSUS ,
                aw.kArt_muscle.GENIOGLOSSUS ,
                aw.kArt_muscle.LUNGS])#.astype('int')  # not actually free but don't want to screw with them
                #aw.kArt_muscle.LEVATOR_PALATINI] # remember to fix levator palatini for any utterance though
free_muscles = np.sort(free_muscles)
print free_muscles
#print free_muscles

dh = DataHandler()

nwin =240 # 30 ms
nstep=80 # 10 ms
overlap = nwin-nstep
for k in range(1, 1000):
    full_dir = directory+str(k)

    print "Loading utterance: ", k
    dh.LoadDataDir(directory=directory+str(k))

    if np.isnan(dh.raw_data['sound_wave']).any():
        print "Utterance contains NaNs"
        print "Skip!"
        print "#"*50
        continue 

    mfcc, mel_spectrum, spectrum = tb.mfcc(dh.raw_data['sound_wave'], nwin=nwin, nstep=nstep, nfft=512, nceps=13, fs=8000)

    #print dh.raw_data['art_hist'].shape
    #print mfcc.shape, mel_spectrum.shape, spectrum.shape

    art_segs = segment_axis(dh.raw_data['art_hist'][free_muscles, :], nwin, overlap, axis=1)

    art_segs = np.swapaxes(art_segs, 0, 1)

    print "Saving I/O data..."
    np.savez("../tensorflow/speech_io/data"+str(k), 
             mfcc=mfcc, mel_spectrum = mel_spectrum, spectrum=spectrum, art_segs=art_segs)
                                                   

    print "Saving animation..."
    dh.SaveAnimation()
    print "Done!"
    print "#"*50

    
    
    #plt.figure()
    #plt.imshow(mfcc)
    #plt.figure()
    #plt.imshow(mel_spectrum)
    #plt.figure()
    #plt.imshow(spectrum)
    #plt.show()
    
