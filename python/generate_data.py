import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
#import primitive.RandomArtword as aw
import Artword as aw
from primitive.RandExcite import RandExcite

import pylab as plt

import argparse

parser = argparse.ArgumentParser(description="Generate data for testing")
parser.add_argument('--round', dest='rnd', type=int, default=1, help="Unique ID number for each data directory")
parser.add_argument('--init', dest='init', default='zeros', help="How to initialize articulation (random or zeros)")
parser.add_argument('--breathe', dest='breathe', default='manual', help="How to initialize articulation (random or manual)")

args = parser.parse_args()
rnd = args.rnd
    
loops = 100
utterance_length = 1.0

# random init
fname = "zero_init_"+str(rnd)
initial_art = np.zeros((aw.kArt_muscle.MAX, ))

if args.init == "random":
    initial_art = np.random.random((aw.kArt_muscle.MAX, ))
    fname = "rand_init_"+str(rnd)

if args.breathe=='manual':
    loops = 10
    fname="breathe_"+fname

directory = "data/test/"+fname

print "Breathing: "+args.breathe
print "Directory: "+directory
print "Loops: "+str(loops)
print "Initial Articulation:"
print initial_art

utterance = RandExcite(directory=directory, 
                       loops=loops, 
                       utterance_length=utterance_length,
                       initial_art=initial_art)

if args.breathe=='manual':
    utterance.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.2], 
                                                          [0.1, 0.0])

utterance.Run(max_increment=0.3, min_increment=0.1, max_delta_target=0.1, random=True, addDTS=False)


