import numpy as np
import scipy.signal as signal
import numpy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler

class SubspaceDFA(DataHandler):

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
        self.Features = None # init to nothing


        # downsample data such that it's some number of ms
        # period at which controller can make observations
        self._observation_period = 1 # in ms

        # Taken care of in parent class.
        #self.home_dir = kwargs.get("home_dir", "../data")

    def SetFeatures(self, feature_class):
        self.Features = feature_class(tubes=self.tubes)

    def LoadRawData(self, raw_data):
        """
        LoadRawData(raw_data)

        Data array is directly passed into the class and copied to the internal _data variable.
        No preprocessing, feature extraction, or resampling is performed on the data loaded 
        using this method.

        """
        # directly pass array into class
        self._data = np.copy(raw_data)

    def LoadDataFile(self, fname, sample_period=8):
        """
        LoadDataFile(sample_period=1)

        Overloaded version of DataHandler.LoadDataFile
        Loads data from file and converts to the appropriate format for subspace method.
        """ 
        #print fname
        self.control_period=sample_period

        # clear internal data dictionary and 
        # load data file according to parent class
        self.data.clear()
        super(SubspaceDFA, self).LoadDataFile(fname)

        _data = self.Features.Extract(self.data, sample_period)

        """
        # TODO: Need to decide what this was intially intended for
        # Disabled because it breaks loading of multiple unrelated files
        if self._data.size==0:
            self._data = _data
        else: 
            self._data = np.append(self._data, _data, axis=1)
        """
        self._data = _data

        # clear out the raw data dictionary to save space
        self.data.clear() 

        return _data

    def PreprocessData(self, past, future, overlap=False, normalize=True):
        # note: axis 0 is parameter dimension, axis 1 is time
        #       sample_period is measured in milliseconds

        # save past, future values for later use
        self._past = past
        self._future = future

        if (len(self._data) == 0) and (len(self.data) == 0):
            print "No data has been loaded."
            return 0

        if normalize:
            self._std = np.std(self._data, axis=1)
            self._ave = np.mean(self._data, axis=1)

            # shift to zero mean and  normalize by standard deviation
            self._data = ((self._data.T-self._ave)/self._std).T

        
        # get dimension of input/output feature space
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

        #gamma_f = ln.cholesky(np.cov(Xf))
        #gamma_p = ln.cholesky(np.cov(Xp))

        [U, S, Vh] = ln.svd(self.F)
        self._U = np.copy(U)
        self._Vh = np.copy(Vh)
        self._S = np.copy(S)

        U = U[:, 0:k]
        # plt.plot(S)
        # plt.show()
        S = np.diag(S[0:k])
        Vh = Vh[0:k, :]

        #K = np.dot(np.dot(np.sqrt(S), Vh), ln.pinv(gamma_p))
        self.K = np.dot(np.sqrt(S), Vh)

        #O = np.dot(np.dot(gamma_f, U), np.sqrt(S))
        self.O = np.dot(U, np.sqrt(S))

    def SavePrimitives(self, fname=None):

        if fname==None:
            fname = 'primitives'

        np.savez(os.path.join(self.home_dir, str(fname)),
                 K=self.K,
                 O=self.O,
                 ave=self._ave,
                 std=self._std,
                 past=self._past,
                 future=self._future, 
                 control_period=self.control_period, 
                 features=self.Features.__class__.__name__, # feature extractor parameters (don't like this way of passing them through)
                 feature_pointer=self.Features.pointer,
                 feature_tubes=self.Features.tubes,
                 control_action=self.Features.control_action)

    def LoadPrimitives(self, fname=None):

        if fname==None:
            fname = 'primitives.npz'

        primitives = np.load(os.path.join(self.home_dir, fname))

        self.K = primitives['K']
        self.O = primitives['O']
        self._ave = primitives['ave']
        self._std = primitives['std']
        self._past = primitives['past'].item()
        self._future = primitives['future'].item()

    def EstimateStateHistory(self, data):
        self.h = np.zeros((self.K.shape[0], data.shape[1]))

        #for t in range(0, self._past):
        #    self.h[:, t] = self.EstimateState(data[:, 0:t])

        for t in range(self._past, data.shape[1]):
            _Xp = np.reshape(data[:, t-self._past:t].T, (-1, 1))
            self.h[:, t] = np.dot(self.K, _Xp).flatten()


if __name__ == "__main__":
    print "Do stuff"
    # Real test: Generate a signal using underlying factors and see if this
    # method infers them
    from features.ArtFeatures import ArtFeatures
    from features.SpectralAcousticFeatures import SpectralAcousticFeatures

    dim = 8

    #down_sample = 5
    #ss = PrimLearn()
    #ss.LoadDataDir('full_random_30')
    #ss.PreprocessData(50, 10, sample_period=down_sample)

    down_sample = 10*8
    ss = SubspaceDFA(home_dir='../data')
    print "loading features class"

    ss.Features = ArtFeatures(tubes=ss.tubes) # set feature extractor
    #ss.SetFeatures(SpectralAcousticFeatures)

    print "loading data dir"
    ss.LoadDataDir('full_random_10', sample_period=down_sample, verbose=True)
    ss.PreprocessData(50, 10)
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

    feature_dim = ss.Features.pointer['all'].size

    for k in range(dim):
        # pull out the kth primitive dimension
        K = ss.K[k,:].reshape(ss._past, feature_dim)
        O = ss.O[:, k].reshape(ss._future, feature_dim)

        _K =((K)+ss._ave) 
        plt.figure();
        for p in range(ss._past):
            plt.plot(_K[p, ss.Features.pointer['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))

        plt.plot( ss._ave[ss.Features.pointer['art_hist']], 'r-')
        plt.title('Articulators Input: '+str(k))

        plt.figure();
        for p in range(ss._past):
            plt.plot(K[p, ss.Features.pointer['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))
        plt.title('Articulators Raw Input: '+str(k))

        #plt.figure()
        #plt.imshow(K[:, ss.features['art_hist']])
        #plt.title('Articulators Input: '+str(k))


        #plt.figure();
        #for p in range(ss._past):
        #    plt.plot(K[p, ss.features['area_function']], 'b-', alpha=1.*(p+1)/(ss._past+1))
        #plt.title('Area Function Input: '+str(k))

        #plt.figure()
        #plt.imshow(K[:, ss.features['area_function']])
        #plt.title('Area Function Input: '+str(k))

        #plt.figure();
        #plt.plot(K[:, ss.features['lung_pressure']], 'b-')
        #plt.title('Lungs Input: '+str(k))
        #plt.figure();
        #plt.plot(K[:, ss.features['nose_pressure']], 'b-')
        #plt.title('Nose Input: '+str(k))

        ##for p in range(ss._past):
        ##    plt.plot(K[p, :], 'b-', alpha=1.*(p+1)/(ss._past+1))
        ##plt.title('Input: '+str(k))

        #plt.show()

    #for# k in range(dim):
        #plt.figure();
        #for f in range(ss._future):
        #    dat = O[f, :]
        #    dat = ((dat.T*ss._std)+ss._ave).T

        #    plt.plot(dat, 'b-', alpha=1.*(ss._future-f+1)/(ss._future+1))
        #plt.title('Output: '+str(k))
    
    plt.show()


