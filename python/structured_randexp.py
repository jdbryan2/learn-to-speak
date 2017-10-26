import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw
from primitive.RandExp import RandExp

import pylab as plt

    
loops = 5
utterance_length = 1.0
full_utterance = loops*utterance_length

rando = RandExp(dirname="full_random_"+str(loops), 
                method="gesture",
                loops=loops,
                utterance_length=utterance_length,
                initial_art=np.zeros((aw.kArt_muscle.MAX, )))

# manually pump the lungs
#rando.SetManualSequence(aw.kArt_muscle.LUNGS,
#                        np.array([0.2, 0.0]),  # targets
#                        np.array([0.0, 0.5]))  # times

rando.Run(max_increment=0.3, min_increment=0.02, max_delta_target=0.2, dirname="full_random_"+str(loops))


# manually open the jaw
jaw_period = 0.5
jaw_period_var = 0.2

jaw_times = np.random.rand(int(np.ceil(full_utterance/(jaw_period-jaw_period_var))))
jaw_times = np.cumsum(jaw_times*jaw_period_var + jaw_period)
jaw_times -= jaw_times[0] 
jaw_times = jaw_times[jaw_times<full_utterance]

jaw_targets = np.random.rand(jaw_times.size)*0.5
jaw_targets[::2] += 0.5
jaw_targets[0] = 0.
#jaw_targets[:, 1] += 0.5
#jaw_targets = jaw_targets.flatten()

jaw_times = np.append(jaw_times, full_utterance)
jaw_targets = np.append(jaw_targets, jaw_targets[-1])
plt.plot(jaw_times, jaw_targets)
plt.show()


rando.SetManualSequence(aw.kArt_muscle.MASSETER,
                        jaw_targets, 
                        jaw_times)

rando.Run(max_increment=0.3, min_increment=0.01, max_delta_target=0.2, dirname="structured_masseter_"+str(loops))
