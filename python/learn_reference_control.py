import os
import numpy as np
import scipy.linalg as la
import pylab as plt

#directory = "data/batch_zeros_100_10"
directory = "data/rand_steps"

index_list = []  # using a list for simplicity
if os.path.exists(directory):
    for filename in os.listdir(directory):
        if filename.startswith('state_action') and filename.endswith(".npz"):
            index_list.append(int(filter(str.isdigit, filename)))


index_list = np.array(sorted(index_list))

is_init = False

for index in index_list:
    print index 
    fname = os.path.join(directory, 'state_action_'+str(index)+'.npz')
    data = np.load(fname)
    if not is_init:
        state = data['state_hist']
        action = data['action_hist']
        is_init = True

    else:
        state = np.append(state, data['state_hist'], axis=1)
        action = np.append(action, data['action_hist'], axis=1)

A = np.dot(state, la.pinv(action))
print A
plt.figure()
plt.imshow(A)
plt.show()

#plt.figure()
#plt.plot(state.T)
#
#plt.figure()
#plt.plot(action.T)

_state = np.dot(A, action)
error = state-_state
print state.shape
plt.plot(error.T)
#for k in range(state.shape[0]):
#    plt.figure()
#    plt.plot(state[k, :], 'b-')
#    plt.plot(_state[k, :], 'r--')
#    plt.show()


#plt.show()


