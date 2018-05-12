import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw
from primitive.RandExcite import RandExcite
from primitive.PrimitiveUtterance import PrimitiveUtterance
from primitive.Utterance import Utterance
from primitive.ActionSequence import ActionSequence
from primitive.DataHandler import DataHandler

import pylab as plt

    
loops = 1
utterance_length = 5. #10.0
full_utterance = loops*utterance_length

savedir = 'data/random_prim_%i'%loops

prim_filename = 'round411'
prim_dirname = 'data/batch_random_12_12'
full_filename = os.path.join(prim_dirname, prim_filename)

prim = PrimitiveUtterance()
prim.LoadPrimitives(full_filename)
prim.utterance = Utterance(directory = savedir, 
                           utterance_length=utterance_length, 
                           loops=loops,
                           addDTS=False)
                           #initial_art = prim.GetControlMean(),

prim.InitializeControl(initial_art = prim.GetControlMean())

sample_period = prim.control_period/prim.utterance.sample_freq
rand = ActionSequence(dim=prim._dim,
                      initial_action=np.zeros(prim._dim),
                      sample_period=sample_period,
                      random=True,
                      min_increment=0.5, # 20*sample_period, 
                      max_increment=0.5, # 20*sample_period,
                      max_delta_target=0.5)

# all factors over 3 to be constant zero
for factor in range(3, prim._dim):
    rand.SetManualTarget(factor, 0., 0.)

handler = DataHandler()
handler.params = prim.GetParams()

for k in range(loops):
    print "Loop %i"%k
    prim.ResetOutputVars()

    while prim.NotDone():
        action = rand.GetAction(prim.NowSecondsLooped())
        prim.SimulatePeriod(control_action=action)

    plt.figure()
    plt.plot(prim.state_hist.T)
    plt.figure()
    plt.plot(prim.action_hist.T)
    plt.show()
    
    prim.SaveOutputs(str(k))

    if k < 10:
        handler.AppendData(prim.GetOutputs())

    prim.LoopBack()

handler.SaveAnimation(directory=savedir)
handler.SaveWav(directory=savedir)


#rando = RandExcite(dirname="random_"+str(loops), 
#                method="gesture",
#                loops=loops,
#                utterance_length=utterance_length,
#                initial_art=np.random.random((aw.kArt_muscle.MAX, )))
#
## manually pump the lungs
##rando.SetManualSequence(aw.kArt_muscle.LUNGS,
##                        np.array([0.2, 0.0]),  # targets
##                        np.array([0.0, 0.5]))  # times
#
#rando.Run(max_increment=0.3, min_increment=0.02, max_delta_target=0.2, dirname="full_random_"+str(loops), addDTS=False)


