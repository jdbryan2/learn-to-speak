# load primitives from file
# control vocal tract 
# generate figures with constant control inputs at various values
# save vocal tract data of utterance to file, and save output audio to file


import os
import numpy as np
import pylab as plt
from primitive.PrimitiveUtterance import PrimitiveUtterance
import Artword as aw
from matplotlib2tikz import save as tikz_save

def PlotTraces(data, rows, max_length, sample_period, highlight=0, highlight_style='b-'):
    #if data.shape[1] < max_length:
    #    max_length= data.shape[1]

    print max_length

    for k in range(rows.size):
        if k == highlight: 
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], highlight_style, linewidth=2)
        else:
            plt.plot(np.arange(max_length)*sample_period/8000., data[rows[k], :max_length], 'b-', alpha=0.3)

    plt.xlim((0, max_length*sample_period/8000.))

#import test_params
from test_params import *
primdir = dirname+'_prim'
ATM = 14696. # one atmosphere in mPSI
ATM = 101325. # one atm in pascals




control = PrimitiveUtterance(dir_name=primdir,
                             prim_fname=load_fname,
                             loops=1,
                             utterance_length=5)

#control.InitializeDir(dirname=primdir, addDTS=False)

print control.K
control.InitializeControl()


while control.speaker.NotDone():
    # do stuff here to choose control action
    current_state = control.SimulatePeriod(control_action=0., control_period=0.)


control.Save()
#plt.plot(_h)
savedir = 'data/' + primdir + '/figures/in_out/'
if not os.path.exists(savedir):
    os.makedirs(savedir)

_h = control.state_hist
print _h.shape

PlotTraces(_h, np.arange(_h.shape[0]), _h.shape[1], sample_period, highlight=0)
plt.grid(True)
tikz_save(savedir + 'state_history.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')

plt.show()

plt.figure()
PlotTraces(control.art_hist, np.arange(control.art_hist.shape[0]), control.art_hist.shape[1], sample_period, highlight=0)
#plt.plot(control.art_hist.T)
plt.show()

