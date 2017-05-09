
import os
import time
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw

# import pylab as plt

################################################################################
# Define simulation parameters
################################################################################
MAX_NUMBER_OF_TUBES = 89
gender = "Female"
sample_freq = 8000  # audio sample frequency
oversamp = 70  # oversampling ratio of kinematics
glottal_masses = 2
utterance_length = 1.0  # seconds
total_loops = 50
max_increment = 0.1  # seconds
min_increment = 0.01  # seconds
max_delta_target = 1.0  # muscle parameter
total_increments = utterance_length/min_increment + 1

# setup target directory
directory = 'episodic_gestures_' + time.strftime('%Y-%m-%d-%H-%M-%S')

if not os.path.exists('data'):
    os.makedirs('data')

directory = 'data/' + directory
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + '/'

################################################################################
# Initialize Simulator
################################################################################
speaker = vt.Speaker(gender, glottal_masses, sample_freq, oversamp)

for iter in range(total_loops):
    print "Loop #" + str(iter)

    ############################################################################
    # Generate random target sequences
    ############################################################################
    # Should simply create an artword class in python

    randart = aw.Artword(utterance_length)
    increments = np.random.random((aw.kArt_muscle.MAX, total_increments)) * \
        (max_increment-min_increment)+min_increment

    time = np.cumsum(increments, 1)

    targets = np.cumsum(
                np.random.random((aw.kArt_muscle.MAX, total_increments))*0.5,
                1) % 1.0

    for k in range(aw.kArt_muscle.MAX):
        for ind, t in enumerate(time[k, :]):

            # set target if it's still in the utterance duration
            if t < utterance_length:
                randart.setTarget(k, targets[k, ind], t)

            # interpolate between current and next target at end of utterance
            else:
                randart.setTarget(
                        k,
                        np.interp(utterance_length, time[k, :], targets[k, :]),
                        utterance_length)
                break  # we've already hit the end of the utterance

    ############################################################################
    # Initialize simulator and run that bitch
    ############################################################################

    articulations = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
    speaker.InitSim(utterance_length, articulations)

    # array that carries the sound waveform
    sound_wave = np.zeros(np.ceil(sample_freq*utterance_length))
    area_function = np.zeros((MAX_NUMBER_OF_TUBES,
                              np.ceil(sample_freq*utterance_length)))
    art_hist = np.zeros((aw.kArt_muscle.MAX,
                         np.ceil(sample_freq*utterance_length)))

    while speaker.NotDone():

        # pass the current articulation in
        randart.intoArt(articulations, speaker.NowSeconds())
        speaker.SetArticulation(articulations)

        speaker.IterateSim()

        # Save sound data point
        sound_wave[speaker.Now()-1] = speaker.GetLastSample()
        speaker.GetAreaFcn(area_function[:, speaker.Now()-1])
        art_hist[:, speaker.Now()-1] = articulations

    scaled = np.int16(sound_wave/np.max(np.abs(sound_wave)) * 32767)
    write(directory + 'audio' + str(iter) + '.wav',
          sample_freq,
          scaled)

    np.savez(directory + 'data' + str(iter),
             sound_wave=sound_wave,
             area_function=area_function,
             art_hist=art_hist,
             gender=gender,
             sample_freq=sample_freq,
             oversamp=oversamp,
             glottal_masses=glottal_masses)

    # import pylab as plt
#
    # for k in range(MAX_NUMBER_OF_TUBES):
    #    # plt.plot(area_func[k, :])
    #    # plt.hold(True)
    # plt.hold(False)
    # plt.show()
