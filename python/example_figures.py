
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib2tikz import save as tikz_save
import scipy.signal as signal

from DataHandler import DataHandler

directory = 'click'
dh = DataHandler()
dh.LoadDataDir(directory)
#dh.SaveAnimation(dirname=directory)

print dh.data.keys()
print dh.data['sound_wave'][0].size

sound = dh.data['sound_wave'][0]
plt.plot(sound)
tikz_save(
    'apa_sound.tex',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth'
    )

f,t,Sxx = signal.spectrogram(sound, 8000)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Freq (Hz)')
plt.xlabel('Time (sec)')


plt.show()
