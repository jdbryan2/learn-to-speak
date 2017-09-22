import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw

from DataHandler import DataHandler

def moving_average(a, n=3, axis=0) :
    ret = np.cumsum(a, axis=axis, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# TODO: create simple class that loads data files, use as parent for prim class

class PrimLearn(DataHandler):

    def __init__(self, **kwargs):

        #super(PrimLearn, self).__init__(**kwargs)
        DataHandler.__init__(self, **kwargs)

        # initialize the variables
        self.Xf = np.array([])
        self.Xp = np.array([])

        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # internal variables for tracking dimensions of past and future histories and internal state dimension
        self._past = 0
        self._future = 0
        self._dim = 0

        # data vars
        self.art_hist = np.array([])
        self.area_function = np.array([])
        self.sound_wave = np.array([])
        self._data = np.array([]) # internal data var

        # Taken care of in parent class.
        #self.home_dir = kwargs.get("home_dir", "../data")

    def LoadRawData(self, raw_data):
        # directly pass array into class
        self._data = np.copy(raw_data)

    def ConvertData(self, sample_period=1):
        # convert data dictionary into useable data array  
        # perform necessary normalizations and trim out unnecessary parts
        # function still not complete...
        if len(self.data) == 0:
            print "No data to convert."
            return 0

        # round down for decimation rate
        #dec_rate = int(np.floor(self.params['sample_freq']/sample_rate))

        # TODO: finish preprocess of data
        # compute moving_average over time dimension and down sample

        # this should include a more detailed process with downsampling
        # and feature extraction

        # loop over indeces and stack them into data array
        art_ave = np.mean(self.data['art_hist'], axis=1)
        art_std = np.std(self.data['art_hist'],axis=1)
        self._data = ((self.data['art_hist'].T-art_ave)/art_std).T

        area_function = self.data['area_function'][self.tubes['all'], :]
        area_ave = np.mean(area_function, axis=1)
        area_std = np.std(area_function, axis=1)

        self._ave = np.append(art_ave, area_ave)
        self._std = np.append(art_std, area_std)
        

        #plt.figure()
        #plt.plot(area_ave)
        #plt.plot(area_ave+area_std, 'r--')
        #plt.plot(area_ave-area_std, 'r--')
        #plt.show()
        
        #print area_function.shape, area_std.shape
        #plt.plot(area_std)
        #plt.show()

        # normalize by standard deviation
        area_function = ((area_function.T-area_ave)/area_std).T
        #area_function  = (area_function.T/area_std).T

        #self._data = np.append(self._data, self.data['area_function'], axis=0)
        self._data = np.append(self._data, area_function, axis=0)
        
        # decimate to 1ms sampling period
        self._data = signal.decimate(self._data, 8, axis=1, zero_phase=True) 
        # decimate to 5ms sampling period
        if sample_period > 1:
            self._data = signal.decimate(self._data, sample_period, axis=1, zero_phase=True)

        #for n in range(len(keys)):
        #    if self._data.size == 0:
        #        self._data = self.data[key[n]]
        #    else:
        #        self._data = np.append(_data, self.data[key[n]], axis=0)


    def PreprocessData(self, past, future, overlap=False, sample_period=1):
        # note: axis 0 is parameter dimension, axis 1 is time
        #       sample_period is measured in milliseconds

        # save past, future values for later use
        self._past = past
        self._future = future

        # check if _data is populated first (data dictionary is ignored if so)
        if len(self._data) == 0:
            self.ConvertData(sample_period)

        if (len(self._data) == 0) and (len(self.data) == 0):
            print "No data has been loaded."
            return 0

        
        dim = self._data.shape[0]

        if overlap==False:
            chunks = int(np.floor(self._data.shape[1]/((past+future))))
            # format data into Xf and Xp matrices
            Xl = self._data[:, :chunks*(past+future)].T.reshape(-1, (past+future)*dim).T  # reshape into column vectors of length past+future
            print Xl.shape
        else:
            print ''
            chunks = int(np.floor(self._data.shape[1]/(past+future)))
            Xl = self._data[:, :(chunks-1)*(past+future):future]
            print Xl.shape
            for k in range(1, past+future):
                Xl = np.append(Xl, self._data[:, k:(chunks-1)*(past+future)+k:future], axis=0)
                print Xl.shape


        self.Xf = Xl[(past*dim):((past+future)*dim), :]
        self.Xp = Xl[0:(past*dim), :]



    def SubspaceDFA(self, k):
        """Decompose linear prediction matrix into O and K matrices"""

        self._dim = k

        # compute predictor matrix
        self.F = np.dot(self.Xf, ln.pinv(self.Xp))

#       #gamma_f = ln.cholesky(np.cov(Xf))
#       #gamma_p = ln.cholesky(np.cov(Xp))

        [U, S, Vh] = ln.svd(self.F)
        self._U = np.copy(U)
        self._Vh = np.copy(Vh)
        self._S = np.copy(S)

        U = U[:, 0:k]
        # plt.plot(S)
        # plt.show()
        S = np.diag(S[0:k])
        Vh = Vh[0:k, :]

#       #K = np.dot(np.dot(np.sqrt(S), Vh), ln.pinv(gamma_p))
        self.K = np.dot(np.sqrt(S), Vh)

#       #O = np.dot(np.dot(gamma_f, U), np.sqrt(S))
        self.O = np.dot(U, np.sqrt(S))

        # return [O, K]

    def EstimateStateHistory(self, data):
        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(ss.K, _Xp).flatten()

    
    # controller functions 
    #######################
    # should set this up so I can feed it area and art individually
    #def EstimateState(self, articulators, area_function): 
    def EstimateState(self, data, normalize=False):
        data_length = data.shape[1]

        if normalize:
            # convert data by mean and standard deviation
            _data = data
            _data = ((_data.T-self._ave)/self._std).T
        
        if data_length < self._past:
            _data = np.append(np.zeros((data.shape[0], self._past-data_length)), _data, axis=1)
            
        _Xp = np.reshape(_data[:, data_length-self._past:data_length].T, (-1, 1))
        current_state = np.dot(ss.K, _Xp).flatten()
        return current_state

    def GetControl(self, current_state):

        _Xf = np.dot(ss.O, current_state)
        _Xf = _Xf.reshape((self._data.shape[0], self._future))
        predicted = _Xf[:, 0]
        predicted = ((predicted.T*self._std)+self._ave).T
        return predicted[0:self.data['art_hist'].shape[0]]


if __name__ == "__main__":
    print "Do stuff"
    # Real test: Generate a signal using underlying factors and see if this
    # method infers them

    dim = 8

    #down_sample = 5
    #ss = PrimLearn()
    #ss.LoadDataDir('full_random_30')
    #ss.PreprocessData(50, 10, sample_period=down_sample)

    down_sample = 10
    ss = PrimLearn()
    ss.LoadDataDir('full_random_100')
    ss.PreprocessData(50, 10, sample_period=down_sample)
    ss.SubspaceDFA(dim)

    ss.EstimateStateHistory(ss._data)
    plt.plot(ss.h.T)
    plt.show()

    #for k in range(dim):
    #    plt.figure();
    #    plt.imshow(ss.K[k, :].reshape(ss._past, 88))
    #    plt.title('Input: '+str(k))

    #for k in range(dim):
    #    plt.figure();
    #    plt.imshow(ss.O[:, k].reshape(ss._future, 88), aspect=2)
    #    plt.title('Output: '+str(k))

    for k in range(dim):
        plt.figure();
        K = ss.K[k,:].reshape(ss._past, 88)
        for p in range(ss._past):
            plt.plot(K[p, :], 'b-', alpha=1.*(p+1)/(ss._past+1))
        plt.title('Input: '+str(k))

    for k in range(dim):
        plt.figure();
        O = ss.O[:, k].reshape(ss._future, 88)
        for f in range(ss._future):
            dat = O[f, :]
            dat = ((dat.T*ss._std)+ss._ave).T

            plt.plot(dat, 'b-', alpha=1.*(ss._future-f+1)/(ss._future+1))
        plt.title('Output: '+str(k))
    
    plt.show()


