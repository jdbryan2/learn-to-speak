
import os
import numpy as np
import pylab as plt
from primitive.SubspacePrim import PrimLearn
from matplotlib2tikz import save as tikz_save

#dim = 8
#sample_period = 10
#dirname = 'full_random_500_prim'
#dirname = 'full_random_500'
#past = 100
#future = 10
from test_params import *

savedir = 'data/' + dirname + '/figures/learn/'
load_fname = dirname + '/primitives.npz' # class points toward 'data/' already, just need the rest of the path

if not os.path.exists(savedir):
    os.makedirs(savedir)

ss = PrimLearn()
ss.ConvertDataDir(dirname, sample_period=sample_period)
ss.PreprocessData(past, future, sample_period=sample_period)
ss.LoadPrimitives(load_fname)

ss.EstimateStateHistory(ss._data)

################################################################################
if ss.h.shape[1] < 1000:
    hist_len = ss.h.shape[1]
else:
    hist_len = 1000
print hist_len
plt.plot(np.arange(ss._past, hist_len)*sample_period/1000., ss.h[:, ss._past:hist_len].T)
plt.xlim((0, hist_len*sample_period/1000.))
plt.xlabel('Time (s)')
plt.ylabel('$h_t$')
tikz_save( savedir+'state_history.tikz',
            figureheight = '\\figureheight',
            figurewidth = '\\figurewidth')


plt.show()

#for k in range(dim):
#    plt.figure();
#    plt.imshow(ss.K[k, :].reshape(ss._past, 88))
#    plt.title('Input: '+str(k))

#for k in range(dim):
#    plt.figure();
#    plt.imshow(ss.O[:, k].reshape(ss._future, 88), aspect=2)
#    plt.title('Output: '+str(k))

feature_dim = ss.features['all'].size

for k in range(dim):
    # pull out the kth primitive dimension
    K = ss.K[k,:].reshape(ss._past, feature_dim)
    O = ss.O[:, k].reshape(ss._future, feature_dim)

    #_K =((K)+ss._ave) 
    #plt.figure();
    #for p in range(ss._past):
    #    plt.plot(_K[p, ss.features['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))

    #plt.plot( ss._ave[ss.features['art_hist']], 'r-')
    #plt.title('Articulators Input: '+str(k))

    #plt.figure();
    #for p in range(ss._past):
    #    plt.plot(K[p, ss.features['art_hist']], 'b-', alpha=1.*(p+1)/(ss._past+1))
    #plt.title('Articulators Raw Input: '+str(k))

    ################################################################################
    # Articulator IN
    plt.figure()
    plt.imshow(K[:, ss.features['art_hist']], interpolation=None, cmap='bone')
    plt.xlabel('Articulator')
    plt.ylabel('Time')

    tikz_save(
        savedir + 'primitive_art_in_'+str(k)+ '.tikz',
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth'
        )

    plt.title('Primitive Input '+str(k))
    plt.close()

    ################################################################################
    # Area IN
    plt.figure()
    plt.imshow(K[:, ss.features['area_function']], interpolation=None, cmap='bone')
    plt.xlabel('Tube')
    plt.ylabel('Time')

    tikz_save(
        savedir+'primitive_area_in_'+str(k)+ '.tikz',
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth'
        )

    plt.title('Primitive Input '+str(k))
    plt.close()


    ################################################################################
    # lung IN
    plt.figure();
    plt.plot(K[:, ss.features['lung_pressure']], 'b-')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    tikz_save(
        savedir+'primitive_lung_in_'+str(k)+ '.tikz',
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth'
        )
    plt.title('Lungs Input: '+str(k))
    plt.close()

    ################################################################################
    # nose IN
    if 'nose_pressure' in ss.features:
        plt.figure();
        plt.plot(K[:, ss.features['nose_pressure']], 'b-')
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        tikz_save(savedir+'primitive_nose_in_'+str(k)+ '.tikz',
                    figureheight = '\\figureheight',
                    figurewidth = '\\figurewidth')
        plt.title('Nose Input: '+str(k))
        plt.close()

    ################################################################################
    # Articulator OUT
    plt.figure()
    plt.imshow(O[:, ss.features['art_hist']], interpolation=None, cmap='bone')
    plt.xlabel('Articulator')
    plt.ylabel('Time')

    tikz_save(
        savedir+'primitive_art_out_'+str(k)+ '.tikz',
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth'
        )

    plt.title('Primitive Output '+str(k))
    #plt.show()
    plt.close()

    ################################################################################
    # Area OUT
    plt.figure()
    plt.imshow(O[:, ss.features['area_function']], interpolation=None, cmap='bone')
    plt.xlabel('Tube')
    plt.ylabel('Time')

    tikz_save(
        savedir+'primitive_area_out_'+str(k)+ '.tikz',
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth'
        )

    plt.title('Primitive Output '+str(k))
    #plt.show()
    plt.close()


    ################################################################################
    # lung OUT
    plt.figure();
    plt.plot(O[:, ss.features['lung_pressure']], 'b-')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    tikz_save(
        savedir+'primitive_lung_out_'+str(k)+ '.tikz',
        figureheight = '\\figureheight',
        figurewidth = '\\figurewidth'
        )
    plt.title('Lungs Output: '+str(k))
    #plt.show()
    plt.close()

    ################################################################################
    # nose OUT
    if 'nose_pressure' in ss.features:
        plt.figure();
        plt.plot(O[:, ss.features['nose_pressure']], 'b-')
        plt.xlabel('Time')
        plt.ylabel('Pressure')
        tikz_save(savedir+'primitive_nose_out_'+str(k)+ '.tikz',
                    figureheight = '\\figureheight',
                    figurewidth = '\\figurewidth')
        plt.title('Nose Output: '+str(k))
        #plt.show()
        plt.close()


plt.show()

