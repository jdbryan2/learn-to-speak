# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
from primitive.SubspaceDFA import SubspaceDFA
from primitive.IncrementalDFA import SubspaceDFA as IncrementalDFA
from features.ArtFeatures import ArtFeatures
from features.SpectralAcousticFeatures import SpectralAcousticFeatures
from primitive.RandExcite import RandExcite
#from primitive.Utterance import Utterance
## pretty sure these aren't used ##
import Artword as aw
import os

#dim = 8
#sample_period = 10
#dirname = 'full_random_500'
#past = 50
#future = 50

import argparse
parser = argparse.ArgumentParser(description="Generate data for testing")
parser.add_argument('--past', dest='past', type=int, default=10)
parser.add_argument('--future', dest='future', type=int, default=10)
parser.add_argument('--period', dest='period', type=int, default=10)
parser.add_argument('--dim', dest='dim', type=int, default=10)
parser.add_argument('--init', dest='init', default='zeros', help="How to initialize articulation (random or zeros)")

args = parser.parse_args()

# call in all the necessary global variables
#from test_params import *
sample_period=args.period*8 # # (*8) -> convert to samples ms
past = args.past
future = args.future
dim = args.dim


def get_last_round(directory):
    index_list = []  # using a list for simplicity
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.startswith('round') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, filename)))

    if len(index_list):
        return max(index_list)
    else:
        return 0

################################################################################
loops = 60 
utterance_length = 1.0
full_utterance = loops*utterance_length
directory = "data/batch_%s_%i_%i"%(args.init, args.past, args.future)

# create save directory if needed
if not os.path.exists(directory):
    print "Creating output directory: " + directory
    os.makedirs(directory)

last_round = get_last_round(directory)

if args.init == 'zeros':
    initial_art=np.zeros((aw.kArt_muscle.MAX, ))
else:
    initial_art=np.random.random((aw.kArt_muscle.MAX, ))
#initial_art[0] = 0.2

rando = RandExcite(directory=(directory+"/sim_params/round%i"%last_round), #method="gesture",
                   loops=loops,
                   utterance_length=utterance_length)
rando.InitializeAll(random=True, max_increment=0.3, min_increment=0.1, max_delta_target=0.2,
                    delayed_start=0, initial_art=initial_art)
                    #delayed_start=np.random.random(), initial_art=initial_art)
# set lungs to always breath out
#rando.SetManualArticulation(aw.kArt_muscle.LUNGS, [0.0, 0.1], [0.2, 0.0])

################################################################################

learn = IncrementalDFA(sample_period=sample_period, past=past, future=future, directory=directory, verbose=True)

if last_round>0:
    print "Loading round %i data..."%last_round
    learn.LoadPrimitives('round'+str(last_round))
    old_F = np.copy(learn.F)
else:
    learn.Features = ArtFeatures()

learn.GenerateData(rando, loops, save_data=False)
learn.SubspaceDFA(dim)
learn.SavePrimitives('round'+str(last_round+1))

if last_round>0:
    error = np.sum(np.abs(old_F - learn.F))
    print "Absolute change to F: %f" % error 



