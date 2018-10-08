

import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.pyraat_vae import VAE, Dynamical_Dataset, variable_summaries
from double_pendulum import Pendulum

LOAD = True
TRAIN = False
EPOCHS = 10
save_dir = './pendunet'
test_name = 'pendunet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

d_train = Dynamical_Dataset('pendulum_train', history_length=10)
d_val = Dynamical_Dataset('pendulum_val', history_length=10)

input_dim, output_dim, state_dim = d_train.parameter_sizes()
latent_size = 3
#print input_size, output_size, state_size 

#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

model = VAE( input_dim, latent_size, state_dim, output_dim, log_dir=log_dir, auto_logging=False, sess=session)

session.run(tf.global_variables_initializer())

#print('Training iteration #{0}'.format(n))


###
if LOAD:
    model.load(load_path)

if TRAIN:
    model.train(d_train, epochs=EPOCHS, batch_size=50, d_val=d_val)
    model.save(save_path)

initial_state = np.radians(np.random.random(4)*360)
step_size = 0.01
pendulum = Pendulum(initial_state=initial_state, step_size=step_size, period=5)

inputs = np.zeros((6,3))
for k in range(3):
    inputs[k,k] = 1.
    inputs[k+3,k] = -1.

for k in range(inputs.shape[0]):
    initial_state = np.radians(np.random.random(4)*360)
    step_size = 0.01
    pendulum = Pendulum(initial_state=initial_state, step_size=step_size, period=5)

    state = np.copy(initial_state)

    output = np.zeros((100, 3))
    for t in range(100):
        actions = model.decode(y=inputs[k, :].reshape((1, 3)), state=state.reshape((1,4)))
        actions = actions.reshape((-1, 2))
        obs = np.zeros(actions.shape)
        
        for n in range(actions.shape[0]):
            state = pendulum.simulate_period(actions[k,:])
            obs[k, :] = pendulum.observed_out()

        output[t, :]  = model.encode(obs.reshape((1, -1)))
    
    plt.plot(output[:, 0])
    plt.plot(output[:, 1])
    plt.plot(output[:, 2])

    plt.show()

    
    print actions.shape

