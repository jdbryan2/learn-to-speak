
# import os
import numpy as np
# from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw

# import pylab as plt

################################################################################
# Define simulation parameters
################################################################################
sample_freq = 8000  # audio sample frequency
oversamp = 70  # oversampling ratio of kinematics
glottal_masses = 2
utterance_length = 1.0  # seconds
total_loops = 10
max_increment = 0.1  # seconds
min_increment = 0.01  # seconds
max_delta_target = 1.0  # muscle parameter
total_increments = utterance_length/min_increment + 1

################################################################################
# Generate random target sequences
################################################################################
randart = aw.Artword(utterance_length)
increments = np.random.random((aw.kArt_muscle.MAX, total_increments)) * \
                (max_increment-min_increment)+min_increment
time = np.cumsum(increments, 1)

targets = np.cumsum(
            np.random.random((aw.kArt_muscle.MAX, total_increments))*0.5,
            1) % 1.0

for k in range(aw.kArt_muscle.MAX):
    print "Starting " + str(k)
    for ind, t in enumerate(time[k, :]):
        print t
        if t < utterance_length:
            randart.setTarget(k, targets[k, ind], t)
        else:
            randart.setTarget(
                        k,
                        np.interp(utterance_length, time[k, :], targets[k, :]),
                        utterance_length)

            print "End of " + str(k)
            print str(ind) + " total targets set"
            break

speaker = vt.Speaker("Female", glottal_masses, sample_freq, oversamp)
