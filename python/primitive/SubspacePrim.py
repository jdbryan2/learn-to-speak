import numpy as np
import numpy.linalg as ln
import os

from DataHandler import DataHandler

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

        # Taken care of in parent class.
        #self.home_dir = kwargs.get("home_dir", "../data")


    def PreprocessData(self, past, future):
        if len(self.data) == 0:
            print "No data has been loaded."
            return 0

        # pull all data from dictionary and into syncronized vectors 

        # this should include a more detailed process with downsampling
        # and feature extraction
        _data = np.array([])
        for key in self.data:
            print self.data[key].shape
            if _data.size == 0:
                _data = self.data[key]
            else:
                _data = np.append(_data, self.data[key], axis=0)

        dim = _data.shape[0]

        # format data into Xf and Xp matrices
        Xl = _data.T.reshape(-1, (past+future)*dim).T  # reshape into column vectors of length 20
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
        # pl.plot(S)
        # pl.show()
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
