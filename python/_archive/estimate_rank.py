# script for learning primitives based on articulator feature outputs
# script generates relevant figures detailing the learning of the primitives
# end of script includes simple control with constant high level input set to 0

import numpy as np
import pylab as plt
import scipy.linalg as lin
from primitive.SubspaceDFA import SubspaceDFA
from features.ArtFeatures import ArtFeatures
from matplotlib2tikz import save as tikz_save
## pretty sure these aren't used ##
#import scipy.signal as signal
#import numpy.linalg as ln
#import os

def AIC(X):
  R = np.cov(X)
  U, s, V = lin.svd(R)

  p, N = X.shape
  R = R/N

  aic = np.zeros(p-1)

  for k in range(p-1):
    aic[k] = 2*k*(2*p-k)

    numerator = 1.
    denominator = 0.
    for i in range(k+1, p):
      numerator *= s[i]**(1./(p-k))
      denominator += s[i]

    #print "Numerator ", k, ":", numerator
    #print "Denominator", k, ":",denominator
    #print "Log:", 2*np.log((p-k)*numerator/denominator), ((p-k)*N)

    aic[k] -= 2*np.log((p-k)*numerator/denominator)*((p-k)*N)

  return aic



dim = 8
sample_period = 8
sample_period_ms = sample_period/8.
dirname = '../data (prelim)/full_random_10'
#dirname = 'structured_masseter_500'

ss = SubspaceDFA(sample_period=sample_period)
ss.Features = ArtFeatures()
#ss.LoadDataDir(dirname)
#ss.ConvertData(sample_period)
ss.LoadDataDir(directory=dirname, sample_period=sample_period)

feature_dim = ss.feature_data.shape[0]

#aic_0 = AIC(ss._data)
#plt.plot(aic_0)
#plt.show()

#print ss.feature_data.shape
print "Data matrix: %i x %i"% (feature_dim, ss.feature_data.shape[1])
max_dim = 20
print "Preprocessing..."
ss.PreprocessData(0, max_dim)
print "Restructured matrix shape: %i x %i"%ss.Xf.shape
print "Computing AIC..."
aic_50 = AIC(ss.Xf)
plt.plot(np.arange(max_dim*feature_dim-1)*1.0*sample_period_ms/feature_dim, aic_50/np.max(aic_50))
plt.xlabel("Time (ms)")
plt.ylabel("AIC")
plt.grid(True)
aic_min = np.argmin(aic_50)
plt.plot([aic_min*1.0*sample_period_ms/feature_dim], [aic_50[aic_min]/np.max(aic_50)], 'ro')

print "Min Value: ", aic_min*1.*sample_period_ms/feature_dim

tikz_save('aic.tikz',
    figureheight = '\\figureheight',
    figurewidth = '\\figurewidth')
plt.show()


#ss.SubspaceDFA(dim)
#
#ss.EstimateStateHistory(ss._data)
#plt.plot(ss.h.T)
#plt.show()
#
#ss.SavePrimitives(dirname+'/primitives')

