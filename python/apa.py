import os
import numpy as np
from scipy.io.wavfile import write
import PyRAAT as vt
import Artword as aw

import pylab as plt


apa = aw.Artword(0.5)
apa.setTarget(aw.kArt_muscle.INTERARYTENOID, 0, 0.5)
apa.setTarget(aw.kArt_muscle.INTERARYTENOID, 0.5, 0.5)
apa.setTarget(aw.kArt_muscle.LEVATOR_PALATINI, 0.0, 1.0)
apa.setTarget(aw.kArt_muscle.LEVATOR_PALATINI, 0.5, 1.0)
apa.setTarget(aw.kArt_muscle.LUNGS, 0.0, 0.2)
apa.setTarget(aw.kArt_muscle.LUNGS, 0.1, 0)
apa.setTarget(aw.kArt_muscle.MASSETER, 0.25, 0.7)
apa.setTarget(aw.kArt_muscle.ORBICULARIS_ORIS, 0.25, 0.2)

sample_freq = 8000
oversamp = 70
glottal_masses = 2

speaker = vt.Speaker("Female", glottal_masses, sample_freq, oversamp)
utterance_length = 0.5

articulations = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))

apa.intoArt(articulations, 0.0)
speaker.InitSim(utterance_length, articulations)

sound_wave = np.zeros(np.ceil(sample_freq*utterance_length))

while speaker.NotDone():

    apa.intoArt(articulations, speaker.NowSeconds())
    speaker.SetArticulation(articulations)

    # alternative way of feeding each muscle individually
    # for k in range(aw.kArt_muscle.MAX):
    #    speaker.setMuscle(k, apa.getTarget(k, speaker.NowSeconds()))

    speaker.IterateSim()

    # sound_wave=np.append(sound_wave, speaker.getLastSample())
    sound_wave[speaker.Now()-1] = speaker.GetLastSample()


plt.plot(sound_wave)
plt.show()

speaker.SaveSound(os.getcwd()+'/apa.txt')

scaled = np.int16(sound_wave/np.max(np.abs(sound_wave)) * 32767)
write('apa1.wav', sample_freq, scaled)

###################
# Looped version

speaker_loop = vt.Speaker("Female", glottal_masses, sample_freq, oversamp)
utterance_length = 0.5
total_loops = 5

# create articulations vector and load values for time 0.0 seconds
articulations = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))
apa.intoArt(articulations, 0.0)

# initialize the simulator
speaker_loop.InitSim(utterance_length/total_loops, articulations)

looped_wave = np.array([])  # np.zeros(np.ceil(sample_freq*utterance_length))

for k in range(total_loops):
    while speaker_loop.NotDone():

        apa.intoArt(articulations, speaker_loop.NowSecondsLooped())
        speaker_loop.SetArticulation(articulations)

        # alternative way of feeding each muscle individually
        # for k in range(aw.kArt_muscle.MAX):
        #    speaker.setMuscle(k, apa.getTarget(k, speaker.NowSeconds()))

        speaker_loop.IterateSim()

        looped_wave = np.append(looped_wave, speaker_loop.GetLastSample())
        # sound_wave[speaker.Now()-1] = speaker.GetLastSample()

    # hit the end of simulator buffer
    speaker_loop.LoopBack()


plt.plot(sound_wave)
plt.hold(True)
plt.plot(looped_wave)
plt.hold(False)
plt.show()

speaker.SaveSound(os.getcwd()+'/apa_looped.txt')

scaled = np.int16(looped_wave/np.max(np.abs(looped_wave)) * 32767)
write('apa2.wav', sample_freq, scaled)
