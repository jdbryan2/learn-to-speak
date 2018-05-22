# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
from primitive.SubspaceDFA import SubspaceDFA
from primitive.IncrementalDFA import SubspaceDFA as IncrementalDFA
from features.ArtFeatures import ArtFeatures
from features.SpectralAcousticFeatures import SpectralAcousticFeatures
#from primitive.RandExcite import RandExcite
from primitive.Utterance import Utterance
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.ActionSequence import ActionSequence
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
parser.add_argument('--period', dest='period', type=int, default=20)
parser.add_argument('--dim', dest='dim', type=int, default=10)
parser.add_argument('--init', dest='init', default='zeros', help="How to initialize articulation (random or zeros)")
#parser.add_argument('--primitive', dest='primitive', default='none', help="Filename of primitive to be loaded")


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
loops = 1#60 
utterance_length = 5.0
full_utterance = loops*utterance_length
directory = "data/mfcc_%s_%i_%i"%(args.init, args.past, args.future)

prim_filename = 'round550'
prim_dirname = 'data/batch_random_1_1'
full_filename = os.path.join(prim_dirname, prim_filename)

prim = PrimitiveUtterance()
prim.LoadPrimitives(full_filename)

# create save directory if needed
if not os.path.exists(directory):
    print "Creating output directory: " + directory
    os.makedirs(directory)

last_round = get_last_round(directory)

if args.init == 'zeros':
    initial_control = np.zeros(prim._dim)
else:
    initial_control = np.random.random(prim._dim)*2. - 1.
#initial_art[0] = 0.2


prim.utterance = Utterance(directory = (directory+"/sim_params/round%i"%last_round), 
                           utterance_length=utterance_length, 
                           loops=loops)

rand = ActionSequence(dim=prim._dim,
                      initial_action=initial_control,
                      sample_period=prim.control_period/8000.,
                      random=True,
                      min_increment=0.1, # 20*sample_period, 
                      max_increment=0.3, # 20*sample_period,
                      max_delta_target=0.8)

prim.InitializeControl(initial_art = prim.GetControl(rand.GetAction(time=0.0)))

prim._act = rand
################################################################################

learn = IncrementalDFA(sample_period=sample_period, past=past, future=future, directory=directory, verbose=True)

if last_round>0:
    print "Loading round %i data..."%last_round
    learn.LoadPrimitives('round'+str(last_round))
    old_F = np.copy(learn.F)
else:
    learn.Features = SpectralAcousticFeatures(control_action='action_hist_1', 
                                                control_sample_period=prim.control_period,
                                                periodsperseg=1) # set feature extractor

learn.GenerateData(prim, loops, save_data=False)
#plt.plot(prim.utterance.data['action_hist_1'].T)
#plt.show()
learn.SubspaceDFA(dim)
#plt.imshow(np.abs(learn.F))
#plt.show()
learn.SavePrimitives('round'+str(last_round+1))

if last_round>0:
    error = np.sum(np.abs(old_F - learn.F))
    print "Absolute change to F: %f" % error 



