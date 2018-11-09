

import numpy as np
import os
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as sig

import tensorflow as tf
from autoencoder.pendulum_vae import VAE, Dynamical_Dataset, variable_summaries
from double_pendulum import Pendulum

LOAD = False
TRAIN = True
EPOCHS = 3
save_dir = './pendunet'
test_name = 'pendunet'
log_dir = save_dir+'/'+test_name+'_logs'
load_path = save_dir+'/'+test_name+'.ckpt'
save_path = save_dir+'/'+test_name+'.ckpt'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

history_length = 1
d_train = Dynamical_Dataset('pendulum_train', history_length=history_length)
d_val = Dynamical_Dataset('pendulum_val', history_length=history_length)


#exit()
input_dim, output_dim, state_dim = d_train.parameter_sizes()
latent_size = 2
#print input_size, output_size, state_size 

#from jh_utilities.datasets.unsupervised_dataset import UnsupervisedDataset
session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#model = VAE( input_dim, latent_size, state_dim, output_dim, log_dir=log_dir, inner_width=20, auto_logging=False, sess=session)
model = VAE( input_dim, latent_size, output_dim, log_dir=log_dir, inner_width=20, auto_logging=False, sess=session)

session.run(tf.global_variables_initializer())

#print('Training iteration #{0}'.format(n))


###
if LOAD:
    model.load(load_path)

if TRAIN:
    model.train(d_train, epochs=EPOCHS, batch_size=100, d_val=d_val)
    model.save(save_path)

initial_state = np.radians(np.random.random(4)*360)
step_size = 0.01
pendulum = Pendulum(initial_state=initial_state, step_size=step_size, period=5)

#inputs = np.zeros((2*latent_size,latent_size))
inputs = np.random.random((100,latent_size))
_outputs = np.zeros((100,latent_size))

#for k in range(latent_size):
    #inputs[k,k] = 1.
    #inputs[k+latent_size,k] = -1.

for k in range(inputs.shape[0]):
    print k
    initial_state = np.zeros(4)#np.radians(np.random.random(4)*360)
    step_size = 0.01
    pendulum = Pendulum(initial_state=initial_state, step_size=step_size, period=5)

    state = np.copy(initial_state)

    output = np.zeros((100, latent_size))
    actions = np.zeros((100, 2))
    for t in range(1):
        actions = model.decode(y=inputs[k, :].reshape((1, latent_size)))#, state=state.reshape((1,4)))
        actions = actions*d_train.action_std+d_train.action_mean
        actions = actions.reshape((-1, 2))
        print actions
        obs = np.zeros(actions.shape)
        
        for n in range(actions.shape[0]):
            state = pendulum.simulate_period(actions[n,:])
            obs[n, :] = pendulum.observed_out()
            obs = (obs-d_train.obs_mean)*d_train.obs_std

        output[t, :]  = model.encode(obs.reshape((1, -1)))
    _outputs[k, :] = np.mean(output, axis=0)
plt.figure()
plt.scatter(_outputs[:, 0], _outputs[:, 1])
    #for k in range(output.shape[1]):
    #    plt.plot(output[:, 0])
    #    plt.plot(output[:, 1])
    #    plt.plot(output[:, 2])

plt.figure()
#plt.scatter(_outputs[:, 0], _outputs[:, 1])
#plt.figure()
#plt.scatter(inputs[:, 0], inputs[:, 1])
plt.plot(inputs)
plt.plot(_outputs)
plt.show()

    

