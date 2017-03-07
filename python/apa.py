import numpy as np
from scipy.io.wavfile import write
import vtSim as vt
import Artword as aw

import pylab as plt

apa = aw.Artword(0.5)
apa.setTarget(aw.kArt_muscle.INTERARYTENOID, 0, 0.5);
apa.setTarget(aw.kArt_muscle.INTERARYTENOID,0.5,0.5);
apa.setTarget(aw.kArt_muscle.LEVATOR_PALATINI,0,1.0);
apa.setTarget(aw.kArt_muscle.LEVATOR_PALATINI,0.5,1.0);
apa.setTarget(aw.kArt_muscle.LUNGS,0,0.2);
apa.setTarget(aw.kArt_muscle.LUNGS,0.1,0);
apa.setTarget(aw.kArt_muscle.MASSETER,0.25,0.7);
apa.setTarget(aw.kArt_muscle.ORBICULARIS_ORIS,0.25,0.2);

sample_freq = 8000;
oversamp = 70;
glottal_masses = 2;

speaker = vt.Speaker("Female", glottal_masses, sample_freq, oversamp)
utterance_length = 0.5

articulations = np.zeros(aw.kArt_muscle.MAX, dtype=np.dtype('double'))

apa.intoArt(articulations, 0.0)

speaker.InitSim(utterance_length, articulations)

sound_wave = np.array([]);
i = 0;
while speaker.NotDone() :

    apa.intoArt(articulations, speaker.NowSeconds())
    speaker.setArticulation(articulations)

    #print speaker.NowSeconds()
    #for k in range(aw.kArt_muscle.MAX):
    #    speaker.setMuscle(k, apa.getTarget(k, speaker.NowSeconds()))

    speaker.IterateSim()

    sound_wave=np.append(sound_wave, speaker.getLastSample())
    i+= 1;
    #if i > 20:
        #break;


plt.plot(sound_wave)
plt.show()

import os
speaker.SaveSound(os.getcwd()+'/apa.txt')

scaled = np.int16(sound_wave/np.max(np.abs(sound_wave)) * 32767)
write('test.wav', sample_freq, scaled)
