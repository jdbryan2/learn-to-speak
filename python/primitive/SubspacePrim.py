import numpy as np
import numpy.linalg as ln
import os
import pylab as plt

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

    def ConvertData(self, sample_rate):
        # convert data dictionary into useable data array  
        # function still not complete...
        if len(self.data) == 0:
            print "No data to convert."
            return 0

        # round down for decimation rate
        dec_rate = int(np.floor(self.params['sample_freq']/sample_rate))
        # TODO: finish preprocess of data
        # compute moving_average over time dimension and down sample

        # this should include a more detailed process with downsampling
        # and feature extraction

        # loop over indeces and stack them into data array
        self._data = self.data['art_hist']
        self._data = np.append(self._data, self.data['area_function'], axis=0)

        #for n in range(len(keys)):
        #    if self._data.size == 0:
        #        self._data = self.data[key[n]]
        #    else:
        #        self._data = np.append(_data, self.data[key[n]], axis=0)


    def PreprocessData(self, past, future, overlap=False, sample_rate=0):
        # note: axis 0 is parameter dimension, axis 1 is time

        # check if _data is populated first (data dictionary is ignored if so)
        if len(self._data) == 0:
            self.ConvertData(sample_rate)

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
        # compute predictor matrix
        self.F = np.dot(self.Xf, ln.pinv(self.Xp))

#       #gamma_f = ln.cholesky(np.cov(Xf))
#       #gamma_p = ln.cholesky(np.cov(Xp))

        [U, S, Vh] = ln.svd(self.F)

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


if __name__ == "__main__":
    print "Do stuff"
    # Real test: Generate a signal using underlying factors and see if this
    # method infers them

    ss = PrimLearn()
    ss.LoadDataDir('apa')
    ss.PreprocessData(10, 10)
