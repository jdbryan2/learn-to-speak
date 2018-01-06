import numpy as np
import scipy.signal as signal
#import numpy.linalg as ln
import scipy.linalg as ln
import os
import pylab as plt
import Artword as aw
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler

class SubspaceDFA(DataHandler):

    def __init__(self, **kwargs):

        super(SubspaceDFA, self).__init__(**kwargs)
        #DataHandler.__init__(self, **kwargs)

        self.InitVars()
        self.DefaultParams()
        self.InitParams(**kwargs)

    def InitVars(self):
        # initialize the variables

        # preprocessed data matrices
        #self.Xf = np.array([])
        #self.Xp = np.array([])

        # intermediate data for computing prediction matrix F without saving all data into memory directly
        # all parameters are saved without any data normalization 
        # mean and variance are computed after (from the sums) and used to adjust phi and psi
        self.phi = np.array([])
        self.psi = np.array([])
        self._sum = np.array([]) # running sum of data
        self._sum2 = np.array([]) # running sum of squared data
        self._count = 0. # total number of data points added to sums

        # primitive model operators
        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        # data vars
        self._data = np.array([]) # internal data var
        self.Features = None # init to nothing
        self.data.clear() # do I want or need this here?

    def DefaultParams(self):
        # internal variables for tracking dimensions of past and future histories and internal state dimension
        self._past = 0
        self._future = 0
        self._dim = 0
        self.sample_period=1
        return 0

    def InitParams(self, **kwargs):

        self._past = kwargs.get('past', self._past)
        self._future = kwargs.get('future', self._future)
        self._dim = kwargs.get('dim', self._dim)
        # period over which data is downsampled
        self.sample_period = kwargs.get('sample_period', self.sample_period)

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

    def LoadDataFile(self, fname, **kwargs):
        """
        LoadDataFile(sample_period=1)

        Overloaded version of DataHandler.LoadDataFile
        Loads data from file and converts to the appropriate format for subspace method.
        """ 
        #print fname
        self.InitParams(**kwargs)

        # clear internal data dictionary and 
        # load data file according to parent class
        ## No longer clearing before we start, leftovers are now important
        #self.data.clear()

        # if soundwave key does not exist
        # create it and fill with zero padding
        if 'sound_wave' not in self.data:
            self.data['sound_wave'] = np.zeros((1, self.Features.min_sound_length))

        # load data and append to self.data dictionary
        super(SubspaceDFA, self).LoadDataFile(fname)
        
        # count number of sample periods in data dictionary
        #total_periods = (self.data['sound_wave'].shape[1] - 
        #                 self.Features.min_sound_length) / self.sample_period
        #total_periods = int(np.floor(total_periods))
        
        _data = self.Features.Extract(self.data, sample_period=self.sample_period)
        # features needs function that returns the size of the used data
        #print total_periods, _data.shape

        if self._data.size==0:
            self._data = _data
        else: 
            self._data = np.append(self._data, _data, axis=1)

        #print _data.shape

        # clear out the raw data dictionary to save space
        #self.data.clear() 
        
        # remove all but the necessary bits
        self.ClearExcessData(size=_data.shape[1]*self.sample_period)

        # does it really matter if we return the data? 
        #return _data
        #self.PreprocessData()

    def ClearExcessData(self, size=None):
        # clear excess data

        # set size to the full length of data if not set
        if size == None: 
            size = self.data['sound_wave'].shape[1]-self.Features.min_sound_length

        # loop over each key in the dictionary and remove all data used to
        # generate features
        for key in self.data:
            if self.Features.min_sound_length > 0 and key=='sound_wave':
                self.data[key] = self.data[key][size:]
            else:
                self.data[key] = self.data[key][:, size:]

    def PreprocessData(self, overlap=False):
        #TODO: Remove option for normalize - assume always true 
        #       Remove from SubspaceDFA.py too.

        # note: axis 0 is parameter dimension, axis 1 is time
        #       sample_period is measured in milliseconds

        # save past, future values for later use
        #self._past = past
        #self._future = future
        if overlap:
            print "Overlap is currently not supported. No overlap used."

        if (len(self._data) == 0) and (len(self.data) == 0):
            print "No data has been loaded."
            return 0

        # count how many chunks of data we can use
        # data_chunks points to the useable chunks   
        # used data removed from self._data
        chunks = int(np.floor(self._data.shape[1]/(self._past+self._future)))
        data_chunks = self._data[:, :chunks*(self._past+self._future)]
        self._data = self._data[:, chunks*(self._past+self._future):]

        # get dimension of feature space
        dim = data_chunks.shape[0]
        print dim,data_chunks.shape


        # Better normalization method is outlined in my notebook: 
        # page 111-113
        self._std = np.std(data_chunks, axis=1)
        self._ave = np.mean(data_chunks, axis=1)

        # shift to zero mean and  normalize by standard deviation
        normed_data_chunks = ((data_chunks.T-self._ave)/self._std).T

        # format data into Xf and Xp matrices
        Xl = normed_data_chunks.T.reshape(-1, (self._past+self._future)*dim).T  # reshape into column vectors of length past+future
        Xf = Xl[(self._past*dim):((self._past+self._future)*dim), :]
        Xp = Xl[0:(self._past*dim), :]
        print Xl.shape

        # don't normalize here, save summations for computing mean and variance
        #  mean -> sum(data_chunks, axis=1)
        #  var -> sum(data_chunks**2, axis=1)
        #  total_count += data_chunks.shape[1]
        if self._count == 0:
            self._sum = np.sum(data_chunks, axis=1)
            self._sum2 = np.sum(np.abs(data_chunks)**2, axis=1)
            self._count = data_chunks.shape[1]
            print "Computing phi"
            self.phi = np.dot(Xf, Xp.T)
            print "Computing psi"
            self.psi = np.dot(Xp, Xp.T)
        else:
            self._sum += np.sum(data_chunks, axis=1)
            self._sum2 += np.sum(np.abs(data_chunks)**2, axis=1)
            self._count += data_chunks.shape[1]
            self.phi += np.dot(Xf, Xp.T)
            self.psi += np.dot(Xp, Xp.T)

        print Xf.shape
        print Xp.shape

        # save summation matrices:
        # sum over Xf and Xp (along dim 1?)
        # sum over Xf**2 and Xp**2
        # save phi=dot(Xf, Xp.T) and psi=dot(Xp, Xp.T)

        # use to compute normalization at start of SubspaceDFA

    def SubspaceDFA(self, k):
        """Decompose linear prediction matrix into O and K matrices"""

        self._dim = k

        # compute mean and variance

        # normalize phi and psi

        # compute predictor matrix
        #self.F = np.dot(self.Xf, ln.pinv(self.Xp))
        print "Computing F"
        print self.psi.shape
        #print ln.matrix_rank(self.phi)
        
        # use scipy.linalg.pinvh to speed up inverting symmetric matrix
        self.F = np.dot(self.phi, ln.pinvh(self.psi))
        print self.F.shape

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
                 control_period=self.sample_period, 
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
        # has to be normalized before processing state history
        data= ((data.T-self._ave)/self._std).T

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


