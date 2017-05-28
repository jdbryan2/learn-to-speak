import numpy as np
import numpy.linalg as ln
import os
x = np.array([])

# TODO: create simple class that loads data files, use as parent for prim class


class PrimLearn:

    def __init__(self, **kwargs):
        # define tube sections
        self.lungs = np.arange(6, 23)
        self.bronchi = np.arange(23, 29)
        self.trachea = np.arange(29, 35)
        self.glottis = np.arange(35, 37)
        self.tract = np.arange(37, 64)
        self.nose = np.arange(64, 78)

        self.all_tubes = self.lungs[:]
        self.all_tubes = np.append(self.all_tubes, self.bronchi)
        self.all_tubes = np.append(self.all_tubes, self.trachea)
        self.all_tubes = np.append(self.all_tubes, self.glottis)
        self.all_tubes = np.append(self.all_tubes, self.tract)
        # nose may only need to be the nasopharangeal port
        self.all_tubes = np.append(self.all_tubes, 64)
        # self.all_tubes = np.append(self.all_tubes, self.nose)

        # initialize the variables
        self.data = np.array([])
        self.Xf = np.array([])
        self.Xp = np.array([])

        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # data vars
        self.art_hist = np.array([])
        self.area_function = np.array([])
        self.sound_wave = np.array([])

        self.home_dir = kwargs.get("home_dir", "data")

    def LoadDataFile(self, fname):
        # load the data from fname, store in class variable
        data = np.load(fname)

        # first index is place in feature vector, second index is time
        self.area_function = data['area_function']
        self.art_hist = data['art_hist']
        self.sound_wave = data['sound_wave']

        if self.data.size == 0:
            self.data = np.copy(self.art_hist)
            self.data = np.append(self.data,
                                  self.area_function[self.all_tubes, :],
                                  axis=0)
        else:
            self.data = np.append(
                self.data,
                np.append(
                    self.art_hist,
                    self.area_function[self.all_tubes, :],
                    axis=0),
                axis=1)

        # extract tubes of interest, compute spectrogram, downsample
        # this should all happen here.
        print "LoadDataFile stub"
        return 0

    def LoadDataDir(self, dirname):
        # open directory, walk files and call LoadDataFile on each
        # is the audio saved in the numpy data? ---> Yes

        # append home_dir to the front of dirname
        dirname = os.path.join(self.home_dir, dirname)

        # pull indeces from the filenames
        index_list = []  # using a list for simplicity
        for filename in os.listdir(dirname):
            if filename.startswith('data') and filename.endswith(".npz"):
                index_list.append(int(filter(str.isdigit, filename)))

        # sort numerically and load files in order
        index_list = sorted(index_list)
        print index_list
        for index in index_list:
            print os.path.join(dirname, 'data'+str(index)+'.npz')
            self.LoadDataFile(os.path.join(dirname, 'data'+str(index)+'.npz'))

        # data in one directory is assumed to be continuous across files
        return 0

    def PreprocessData(self, past, future):
        if self.data.size == 0:
            print "No data has been loaded."
            return 0

        # format data into Xf and Xp matrices
        dim = self.data.shape[0]

        Xl = self.data.T.reshape(-1, (past+future)*dim).T  # reshape into column vectors of length 20
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
    ss.PreprocessData()
    ss.LoadDataDir('gesture_2017-05-10-20-16-18')
