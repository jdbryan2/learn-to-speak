import os
import numpy as np
import scipy.linalg as la
import pylab as plt
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct

#directory = "data/batch_zeros_100_10"
#directory = "data/rand_steps_full"
directory = "data/rand_steps_full"

index_list = []  # using a list for simplicity
if os.path.exists(directory):
    for filename in os.listdir(directory):
        if filename.startswith('state_action') and filename.endswith(".npz"):
            index_list.append(int(filter(str.isdigit, filename)))


index_list = np.array(sorted(index_list))

is_init = False
A_dim = 5 


for index in index_list:
    print index 
    fname = os.path.join(directory, 'state_action_'+str(index)+'.npz')
    data = np.load(fname)
    if not is_init:
        print data['action_hist'][:, 0], data['state_hist'][:, 0]
        print data['action_hist'][:, -1], data['state_hist'][:, -1]
        state = data['state_hist'][:A_dim, :]
        action = data['action_hist'][:A_dim, :]

        state_fl = data['state_hist'][:A_dim, 0].reshape((A_dim, 1))
        state_fl = np.append(state_fl, data['state_hist'][:A_dim, -1].reshape((A_dim, 1)), axis=1)
        action_fl = data['action_hist'][:A_dim, 0].reshape((A_dim, 1))
        action_fl = np.append(action_fl, data['action_hist'][:A_dim, -1].reshape((A_dim, 1)), axis=1)
        is_init = True

    else:
        state = np.append(state, data['state_hist'][:A_dim, :], axis=1)
        action = np.append(action, data['action_hist'][:A_dim, :], axis=1)

        state_fl = np.append(state_fl, data['state_hist'][:A_dim, 0].reshape((A_dim, 1)), axis=1)
        state_fl = np.append(state_fl, data['state_hist'][:A_dim, -1].reshape((A_dim, 1)), axis=1)
        action_fl = np.append(action_fl, data['action_hist'][:A_dim, 0].reshape((A_dim, 1)), axis=1)
        action_fl = np.append(action_fl, data['action_hist'][:A_dim, -1].reshape((A_dim, 1)), axis=1)

mlpr = MLPRegressor(hidden_layer_sizes=(200,), 
                    activation="relu", 
                    solver='lbfgs', 
                    max_iter=1000)

train = 2000 #state.shape[1]-50#int(0.9*state.shape[1])
test = 50
state_train = state[:, :train]
state_test  = state[:, -test:]
action_train = action[:, :train]
action_test  = action[:, -test:]

mlpr.fit(action_fl.T, state_fl.T)
mlpr.fit(action_train.T, state_train.T)

#action_test = data['action_hist'][:A_dim, :]
#state_test = data['state_hist'][:A_dim, :]
#state_pred = mlpr.predict(action_test.T)

#svr = SVR(kernel='rbf', gamma=0.1, cache_size=100)
#svr.fit(action_fl.T, state_fl[0, :].T)

kernel = RBF(10)+WhiteKernel(1e-1) #+ DotProduct()#, (1e-3, 1e2))#, (1e-3, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gpr.fit(action_fl.T, state_fl.T)
state_pred = gpr.predict(action_test.T)



for k in range(A_dim):
    plt.plot(state_test[k, :], 'b-')
    plt.plot((state_pred.T)[k, :], 'r--')

plt.show()
