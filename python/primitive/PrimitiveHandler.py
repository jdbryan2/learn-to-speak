import numpy as np
import scipy.signal as signal
import scipy.linalg as ln
import os
import pylab as plt
from matplotlib2tikz import save as tikz_save

from DataHandler import DataHandler
from BaseObject import BaseObject


class PrimitiveHandler(BaseObject):

    def __init__(self, **kwargs):
        """ Class initialization

        Arguments: 

        Outputs:
            N/A

        """

        super(PrimitiveHandler, self).__init__(**kwargs)

        # run initialization if this is the top level class
        if type(self) == PrimitiveHandler:
            self.InitVars()
            self.DefaultParams()
            self.UpdateParams(**kwargs)

    def InitVars(self):
        """ Initialize internal variables

        Arguments: 
            N/A

        Outputs:
            N/A

        """
        # initialize the variables
        super(PrimitiveHandler, self).InitVars()

        # intermediate data for computing prediction matrix F without saving all data into memory directly
        # all parameters are saved without any data normalization 
        # mean and variance are computed after (from the sums) and used to adjust phi and psi
        self.phi = np.array([])
        self.psi = np.array([])
        self._sum = np.array([]) # running sum of data
        self._sum2 = np.array([]) # running sum of squared data
        self._count = 0. # total number of data points added to sums

        self._mean = 0.
        self._2nd_moment = 1.
        self._std = 1.

        # primitive model operators
        self.F = np.array([])
        self.O = np.array([])
        self.K = np.array([])

        self._sample_period = 1

        # feature extractor class
        self.Features = None # init to nothing

    def DefaultParams(self):
        """ Set default parameters of subspace method

        Arguments: 
            N/A

        Outputs:
            N/A

        """
        super(PrimitiveHandler, self).DefaultParams()

        self.directory="data" # default directory name data/utterance_<current dts>

        # internal variables for tracking dimensions of past and future histories and internal state dimension
        self._past = 0
        self._future = 0
        self._dim = 0
        self._sample_period = 1
        self._verbose = True

        self._downpointer_fname      = None
        self._downpointer_directory  = None
        return 0

    def UpdateParams(self, **kwargs):
        """ Parameter initialization

        Arguments: 
            past -- number of past samples used in history
            future -- number of future predicted samples
            dim -- internal dimension of primitive space
            sample_period -- period of downsampling operator

        Outputs:
            N/A

        """
        super(PrimitiveHandler, self).UpdateParams(**kwargs)

        self._past = kwargs.get('past', self._past)
        self._future = kwargs.get('future', self._future)
        self._dim = kwargs.get('dim', self._dim)
        self._verbose = kwargs.get('verbose', self._verbose)
        # period over which data is downsampled
        self._sample_period = kwargs.get('sample_period', self._sample_period)

        self._downpointer_fname = kwargs.get('downpointer_fname', self._downpointer_fname)
        self._downpointer_directory = kwargs.get('downpointer_directory', self._downpointer_directory)

        self.directory = kwargs.get("directory", self.directory)

    def SavePrimitives(self, fname=None, directory=None):
        """Save primitive operators to file

        Arguments: 
            fname -- output file name, default to 'primitives'

        Outputs:
            N/A
            
        Affected Attributes: 
            N/A

        """

        if directory != None: 
            self.directory = directory

        if fname==None:
            fname = 'primitives'

        kwargs = {}
        # incremental DFA parameters
        kwargs['phi'] = self.phi
        kwargs['psi'] = self.psi
        kwargs['count'] = self._count
        kwargs['var'] = self._var
        kwargs['sum'] = self._sum
        kwargs['sum_f'] = self._sum_f
        kwargs['sum_p'] = self._sum_p
        kwargs['count_fp'] = self._count_fp

        # final DFA parameters
        kwargs['F']=self.F
        kwargs['K']=self.K
        kwargs['O']=self.O
        kwargs['mean']=self._mean
        kwargs['std']=self._std
        kwargs['past']=self._past
        kwargs['future']=self._future
        kwargs['sample_period']=self._sample_period
        kwargs['features']=self.Features # Feature extractor class

        # pointers to lower level primitives
        kwargs['downpointer_fname']=self._downpointer_fname     
        kwargs['downpointer_directory']=self._downpointer_directory 

        # create save directory if needed
        if not os.path.exists(self.directory):
            if self._verbose:
                print "Creating output directory: " + self.directory
            os.makedirs(self.directory)

        print kwargs
        np.savez(os.path.join(self.directory, fname), **kwargs)

    def LoadPrimitives(self, fname=None, directory=None):
        """Load primitive operators from file

        Arguments: 
            fname -- input file name, default to 'primitives.npz'

        Outputs:
            N/A
            
        Affected Attributes: 
            N/A

        """

        if directory != None: 
            self.directory = directory

        if fname==None:
            fname = 'primitives.npz'

        if fname[-4:] != ".npz":
            fname += ".npz"

        self.fname = fname

        primitives = np.load(os.path.join(self.directory, fname))

        # incremental DFA parameters
        self.phi = primitives['phi']      
        self.psi = primitives['psi']      
        self._count = primitives['count']    
        self._var = primitives['var']      
        self._sum = primitives['sum']      
        self._sum_f = primitives['sum_f']    
        self._sum_p = primitives['sum_p']    
        self._count_fp = primitives['count_fp'] 

        self.F = primitives['F']
        self.K = primitives['K']
        self.O = primitives['O']
        self._mean = primitives['mean']
        self._std = primitives['std']
        self._past = primitives['past'].item()
        self._future = primitives['future'].item()

        self._dim = self.K.shape[0]

        # Fucking backward compatibility
        if 'sample_period' in primitives:
            self._sample_period = primitives['sample_period'].item() 
        elif 'control_period' in primitives:
            self._sample_period = primitives['control_period'].item() 

        self.Features = primitives['features'].item() # load's feature class 

        # only load pointers down to 
        if 'downpointer_fname' in primitives:
            print "downpointer fname", primitives['downpointer_fname'].item()
            self._downpointer_fname     = primitives['downpointer_fname'].item()
        else: 
            self._downpointer_fname      = None

        if 'downpointer_directory' in primitives:
            print "downpointer directory", primitives['downpointer_directory'].item()
            self._downpointer_directory = primitives['downpointer_directory'].item()
        else: 
            self._downpointer_directory  = None

    def EstimateStateHistory(self, data): # data is returned by FeaturesExtract
        """Estimate primitive state history from data

        Arguments: 
            data -- input data matrix (after feature extraction and downsampling)

        Outputs:
            N/A
            
        Affected Attributes: 
            h -- primitive state history array, first dim is primitive index, second is time

        """

        # has to be normalized before processing state history
        _data = ((data.T-self._mean)/self._std).T

        h = np.zeros((self.K.shape[0], _data.shape[1]))

        for t in range(self._past, _data.shape[1]):
            _Xp = np.reshape(_data[:, t-self._past:t].T, (-1, 1))
            h[:, t] = np.dot(self.K, _Xp).flatten()

        return h

    def EstimateControlHistory(self, data):

        _data = ((data.T-self._mean)/self._std).T

        action_pointer= self.Features.pointer[self.Features.control_action]

        #O_inv = ln.pinv(self.O)[:, action_pointer]
        O_inv = ln.pinv(self.O[action_pointer, :]) # this is the correct formulation

        v = np.dot(O_inv, _data[action_pointer, :])

        return v
