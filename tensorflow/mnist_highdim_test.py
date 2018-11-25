
import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import tensorflow as tf
from autoencoder.vae import VAE, MNIST_Dataset, variable_summaries, null_distortion, blur_distortion

# Flags for saving time
LOAD = False  # load a previsou model at the location of load_path
TRAIN = True # train the model for another round of epochs
BLUR = False # whether or not to use the blurring distortion
EPOCHS = 20

# helper functions

# wrapper for showing mnist vectors as images
def mnist_show(x):
    plt.imshow(x.reshape((28, 28)))


if BLUR:
    test_name = 'mnist_blur_hd'
    distortion = blur_distortion  # function pointer
else:
    test_name = 'mnist_null_hd'
    distortion = null_distortion # function pointer

save_dir = './trained/mnist_hd'

#log_dir = '/home/jacob/Projects/Data/vae/mnist-test'
#log_dir = '/home/jbryan/Data/vae-test'
save_path = save_dir+'/'+test_name+'.ckpt'
load_path = save_dir+'/'+test_name+'.ckpt'
log_dir = save_dir+'/'+test_name+'_logs'

# make the save and log paths if they don't already exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    LOAD = False # can't load anything if we just created the directory we'd load from

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

mnist_path = '/home/jacob/Projects/Data/MNIST_data'
#mnist_path = '/home/jbryan/mnist' # badger path

input_dim = 28*28
output_dim = 28*28
latent_size = 10 



#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, output_dim, netwidth=50, lr=1e-3, log_dir=log_dir, auto_logging=False, sess=session)

session.run(tf.global_variables_initializer())

#print('Training iteration #{0}'.format(n))
d_train = MNIST_Dataset(mnist_path, distortion)
d_val = MNIST_Dataset(mnist_path, distortion, train=False)


###
if LOAD:
    model.load(load_path)

if TRAIN:
    model.train(d_train, epochs=EPOCHS, batch_size=50, d_val=d_val)
    model.save(save_path)



num_points = 200
tx = (0.5-np.random.random((num_points, latent_size)))*2.
img_out = model.decode(tx)
img_in = distortion(img_out)
rx = model.encode(img_out)
rx_std = model.encode_std(img_out)

error = np.abs(tx-rx)
#print(error)
print 'Average transmission error: ', np.mean(error)

#for k in range(tx.shape[0]):
for k in range(10):
#    plt.figure()
#    plt.plot(tx[k])
#    plt.plot(rx[k], 'r')
#    plt.plot(rx[k]+rx_std[k], 'r--')
#    plt.plot(rx[k]-rx_std[k], 'r--')
    plt.figure()
    mnist_show(img_in[k])
    plt.figure()
    mnist_show(img_out[k])
    
    plt.show()





img_in, _ = d_val.get_batch(num_points)
img_in = distortion(img_in)
encoded = model.encode(img_in)
img_out = model.decode(encoded)

re_encoded = model.encode(img_out)

error = np.abs(encoded-re_encoded)

# visualize some outputs

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
ax.scatter(encoded[:, 0], encoded[:, 1], encoded[:, 2], c='b', marker='^')
ax.scatter(re_encoded[:, 0], re_encoded[:, 1], re_encoded[:, 2], c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()



#print encoded
