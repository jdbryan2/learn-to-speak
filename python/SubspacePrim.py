import numpy as np
#import os
x = np.array([])


class PrimLearn:

    def __init__(self, **kwargs):
        # initialize the variables
        self.Xf = np.array([])
        self.Xp = np.array([])

        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        self.data = np.array([])  # {}

        self.home_dir = kwargs.get("home_dir", "data")

    def LoadDataFile(self, fname):
        # load the data from fname, store in class variable
        self.data = np.load(fname)

        # is there any value in having this function?
        # this function should load the data and attach it to Xp & Xf
        # through whatever windowing method is prescribed

        # extract tubes of interest, compute spectrogram, downsample
        # this should all happen here.
        print "LoadDataFile stub"
        return 0

    def LoadDataDir(self, dirname):
        print "LoadDataDir stub"
        # open directory, walk files and call LoadDataFile on each
        # is the audio saved in the numpy data? ---> Yes

        # data in one directory is assumed to be continuous across files
        # all data from the dir will be concatenated and stacked into Xp, Xf
        return 0

    def PreprocessData(self):
        if not self.data:
            print "No data has been loaded."
            return 0

        # format data into Xf and Xp matrices


    def SubspaceDFA(self, k):
        """Decompose linear prediction matrix into O and K matrices"""
        # compute predictor matrix
        F = np.dot(Xf, ln.pinv(Xp))

#       #gamma_f = ln.cholesky(np.cov(Xf))
#       #gamma_p = ln.cholesky(np.cov(Xp))

        [U, S, Vh] = ln.svd(F)

        U = U[:, 0:k]
        pl.plot(S)
        pl.show()
        S = np.diag(S[0:k])
        Vh = Vh[0:k, :]

#       #K = np.dot(np.dot(np.sqrt(S), Vh), ln.pinv(gamma_p))
        K = np.dot(np.sqrt(S), Vh)

#       #O = np.dot(np.dot(gamma_f, U), np.sqrt(S))
        O = np.dot(U, np.sqrt(S))

        return [O, K]


if __name__ == "__main__":
    print "Do stuff"
