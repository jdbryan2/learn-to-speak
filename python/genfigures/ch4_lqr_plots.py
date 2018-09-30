# Script for evaluating primitives by looking at distributions of
# ipa utterances in the low-dimensional space.

from primitive.DataHandler import DataHandler
from features.ArtFeatures import ArtFeatures
import matplotlib.pyplot as plt
import numpy as np

from plot_functions import *

data_dir = '../data'
#dirname = '../data/batch_random_20_5'
#dirname = '../data/batch_random_1_1'
dirname = data_dir+'/batch_random_20_10'

ind= get_last_index(dirname, 'round')
load_fname = 'round%i.npz'%ind

# create image plots of A and B matrices
feedback = np.load(dirname+'/feedback.npz')

plt.figure()
plt.imshow(feedback['A'], interpolation='none')
tikz_save('tikz/state_transition.tikz')

plt.figure()
plt.imshow(np.abs(feedback['B']), interpolation='none')
tikz_save('tikz/control_input.tikz')

plt.show(block=False)

# create plots of state trajectories 
target_data = np.load(dirname+'/500_out_zero/target_data.npz')
sample_period_ms = 5
sample_period = sample_period_ms*8

dh = DataHandler()
dh.LoadDataDir(directory=dirname+'/500_out_zero')
state_hist =dh.raw_data['state_hist_1'] 
action_hist = dh.raw_data['action_hist_1']
art_hist = dh.raw_data['art_hist']
art_hist = art_hist[:, ::sample_period]

plt.figure()
PlotTraces(state_hist, np.arange(state_hist.shape[0]), state_hist.shape[1], sample_period, highlight=3)
plt.xlabel('Time (s)')
plt.ylabel('State Value')
tikz_save('tikz/control_states.tikz')


plt.figure()
PlotTraces(art_hist, np.arange(art_hist.shape[0]), art_hist.shape[1], sample_period, highlight=19)
plt.xlabel('Time (s)')
plt.ylabel('Articulator Value')
tikz_save('tikz/control_arts.tikz')




# create plots of articulator trajectories

plt.show()

