import numpy as np
import scipy.signal as signal
from scipy.fftpack import dct

import pylab as plt
from scipy.io.wavfile import read as wav_read

def Freq2Mel(freq):
    return 2595. * np.log10(1.+freq/700.)

def Mel2Freq(mel):
    return 700. * (10.**(mel/2595.0)-1.)

def PreemphasisFilter(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def MelFilters(nbins, nfft, sample_freq, low_freq=0, high_freq=None):
    # largely borrowed from python_speech_features by James Lyons

    # set upper frequency
    high_freq = high_freq or sample_freq/2. # clever use of 'or' 
    assert high_freq <= sample_freq/2., "high_freq is greater than sample_freq/2"

    # compute mel bins (evenly spaced on mel scale)
    low_mel = Freq2Mel(low_freq)
    high_mel = Freq2Mel(high_freq)
    mel_bins = np.linspace(low_mel, high_mel, nbins+2)
    #print mel_bins

    fft_bins = np.floor((nfft+1)*Mel2Freq(mel_bins)/sample_freq)
    #print fft_bins

    # note: the '//' operator is divides and truncates to integer value
    filter_bank = np.zeros([nbins, nfft//2+1])
    for j in range(0, nbins):

        for i in range(int(fft_bins[j]), int(fft_bins[j+1])):
            filter_bank[j, i] = (i-fft_bins[j])/(fft_bins[j+1]-fft_bins[j])

        for i in range(int(fft_bins[j+1]), int(fft_bins[j+2])):
            filter_bank[j, i] = (fft_bins[j+2]-i)/(fft_bins[j+2]-fft_bins[j+1])

    return filter_bank

def MFCC(data, ncoeffs, nfilters, nfft=512, sample_freq=16000, low_freq=300,
         high_freq=None, preemph = 0.95, window='hamming', nperseg=512,
         noverlap=0):


    data = PreemphasisFilter(data, preemph)

    filter_bank = MelFilters(nfilters, nfft, sample_freq, low_freq, high_freq)

    f, t, spectrum = signal.stft(data, 
                                 fs=sample_freq, 
                                 window=window, 
                                 nperseg=nperseg,
                                 noverlap=noverlap,
                                 nfft=nfft)

    #print spectrum.shape

    #spectrum = spectrum[:, :, :-1] # trim off last element
    #spectrum = spectrum.reshape(spectrum.shape[1], spectrum.shape[2]) # remove first dim
    spectrum = np.abs(spectrum)**2. # convert to power spectrum
    print spectrum.shape

    energy = np.sum(spectrum,1)
    energy = np.where(energy == 0,np.finfo(float).eps,energy)

    features = np.dot(filter_bank, spectrum)
    features = np.where(features == 0,np.finfo(float).eps,features)

    features = features.T

    features = np.log(features)
    #print features.shape

    # I don't really know why the ortho paramter is important
    # it applies a scaling factor but no explanation is given in scipy docs
    features = dct(features, type=2, axis=1, norm='ortho')[:, :ncoeffs]

    return features, energy

def Delta(data):
    output = np.copy(data)
    output[1:, :] -= data[:-1, :]
    output[-1, :] = 0.
    output[0, :] = 0.
    return output


 
def DynamicProgramming(x, y):
    # x, y - input data matrices
    #        first dimension is time, second is feature

    # Dynamic programming function
    # Copy and paste from Matlab 
    lattice = np.ones((x.shape[0], y.shape[0]))*np.infty
    backpointers = np.zeros((x.shape[0], y.shape[0]))
    if x.shape[0] > 2.*y.shape[0] or y.shape[0] > 2.*x.shape[0]:
        distance = np.infty
        return distance, lattice, backpointers

    # loop over x axis of lattice
    for i in range(0, x.shape[0]):

        # compute the global constraints
        y_min = np.ceil(np.max([0.5*i, y.shape[0]-2*(x.shape[0]-i)]))
        y_max = np.floor(np.min([2.*i, y.shape[0]-0.5*(x.shape[0]-i)]))
        #y_min = 0
        #y_max = y.shape[0]-1

        #if i==0:
        #    print "first constraints"
        #    print y_min, y_max

        # loop over y axis of lattice within global constrains
        for j in range(int(y_min), int(y_max)+1):

            # impose local constraints (no more than two steps back)
            k=3
            if j < 3:
                k = j

            # compute optimal distance to each point
            if i > 0:
                min_val = np.min(lattice[i-1, (j-k):(j+1)])
                backpointer = j-k+np.argmin(lattice[i-1, (j-k):(j+1)])
                #min_val = np.min(lattice[i-1, :])
                #backpointer = j-k+np.argmin(lattice[i-1, :])
            else:
                min_val = 0.
                backpointer = 0

            #min_val=0

            #plt.figure()
            #plt.plot(x[i])
            #plt.plot(y[j])
            #plt.show()
            lattice[i, j] = np.sqrt(np.sum(np.abs(x[i]-y[j])**2.)) #/ \
                            #np.sum(np.abs(x[i])**2.)/np.sum(np.abs(y[j])**2.))+ \
            lattice[i, j] += min_val
            backpointers[i, j] = backpointer
        #plt.plot(lattice[i, :])
        #plt.show()


    distance = lattice[i,j] / np.min([x.shape[0], y.shape[0]])
    return distance, lattice, backpointers


if __name__ == '__main__':
    #import pylab as plt
    #from scipy.io.wavfile import read as wav_read

    # Spoken Language Processing: only first 13 MFCC needed for speech recog
    #x = MelFilters(13, 512, 16000, 300., 8000.)
    #for filt in x:
        #plt.plot(filt)

    rate, data = wav_read('/home/jacob/Projects/learn-to-speak/analysis/manifolds/timit/sa1.wav')
    data = 1.0*data/(2**15) # convert from 16 bit integer encoding to [-1, 1]
    
    #print rate, max(data), min(data)
    #plt.figure()
    #plt.plot(data)
    #plt.show()

    #y, e = MFCC(np.random.random(2000), 10)
    y, e = MFCC(data, ncoeffs=13, nfilters=26)
    y2, e = MFCC(data, ncoeffs=13, nfilters=26, nfft=512, nperseg=512,
                 noverlap=128)

    #print y.shape
    #plt.imshow(np.abs(y[:, 1:].T))
    #plt.figure()
    #plt.imshow(np.abs(y2[:, 1:].T))
    #plt.show()

    #distance, lattice, backpointers, constraint = DynamicProgramming(y, y2)

    nperseg=160 # 20 ms * 8 samples per ms
    noverlap=int(3*nperseg/4)

    distance = np.ones((100, 100))*np.infty
    for i in range(0,100):
        fname = "d%02i.wav"%i
        print "Comparing "+fname
        rate, data = wav_read('../data/digits/'+fname)
        data = 1.0*data/(2**15) # convert from 16 bit integer encoding to [-1, 1]

        x, e = MFCC(data, ncoeffs=13, nfilters=26, nfft=512, nperseg=nperseg,
                     noverlap=noverlap, sample_freq=rate)

        delta_x = Delta(x)
        delta_delta_x = Delta(delta_x)
        x = np.append(x, delta_x, axis=1)
        x = np.append(x, delta_delta_x, axis=1)

        for j in range(0, 100):
            fname = "d%02i.wav"%j
            rate, data = wav_read('../data/digits/'+fname)
            data = 1.0*data/(2**15) # convert from 16 bit integer encoding to [-1, 1]

            y, e = MFCC(data, ncoeffs=13, nfilters=26, nfft=512, nperseg=nperseg,
                     noverlap=noverlap, sample_freq=rate)

            delta_y = Delta(y)
            delta_delta_y = Delta(delta_y)
            y = np.append(y, delta_y, axis=1)
            y = np.append(y, delta_delta_y, axis=1)
            d, l, b = DynamicProgramming(x[:, 1:],y[:, 1:])
            d2, l2, b2 = DynamicProgramming(y[:, 1:],x[:, 1:])
            print "Distance Difference: %02i vs %02i" % (i, j)
            d = (d+d2)/2.
            if d < distance[i, j]:
                distance[i, j] = d 


    plt.imshow(distance, cmap='jet')
    plt.show()
