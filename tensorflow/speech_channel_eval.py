import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

from genfigures.plot_functions import *


save_dir = './figures'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


latent_size = 7
#print input_size, output_size, state_size 

data = np.load('3D_results.npz')
h_in = data['coords']
h_out = data['coords_out']
mse = data['mean_error']

print np.min(mse), np.max(mse), np.mean(mse), mse.shape

ind = np.argsort(mse)

#plt.plot(mse[ind])

single= h_in[0]
print single

distances = np.mean(np.abs(h_in - single)**2, axis=1)

print np.min(distances), np.max(distances), np.mean(distances), distances.shape

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
#ax.scatter(h_in[ind[:100], 0], h_in[ind[:100], 1], h_in[ind[:100], 2], c='b', marker='^')
ax.scatter(h_out[ind, 0], h_out[ind, 1], h_out[ind, 2], c='r', marker='^')
#ax.scatter(h_in[ind[-100:], 0], h_in[ind[-100:], 1], h_in[ind[-100:], 2], c='r', marker='^')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
