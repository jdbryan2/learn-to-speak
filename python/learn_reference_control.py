import os
import numpy as np
import scipy.linalg as la
import pylab as plt

#directory = "data/batch_zeros_100_10"
directory = "data/rand_steps_full"

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
        state_0 = np.copy(data['state_hist'][:, :-1])
        state_1 = data['state_hist'][:, 1:]
        action_1 = data['action_hist'][:, 1:]
        is_init = True

    else:
        state_0 = np.append(state_0, data['state_hist'][:, :-1], axis=1)
        state_1 = np.append(state_1, data['state_hist'][:, 1:], axis=1)
        action_1 = np.append(action_1, -data['action_hist'][:, :-1] + data['action_hist'][:, 1:], axis=1)
        #action_1 = np.append(action_1, data['action_hist'][:, 1:], axis=1)

AB = np.dot(state_0, la.pinv(np.append(state_1, action_1, axis=0)))
print AB

A = AB[:, :10]
B = AB[:, 10:]
plt.figure()
plt.imshow(A)
plt.figure()
plt.imshow(B)
plt.show()


state = data['state_hist'][:, 1:]
action = -data['action_hist'][:, :-1] + data['action_hist'][:, 1:]
#state = data['state_hist']
#action = data['action_hist']
_state = np.dot(A, state)
_state += np.dot(B, action)
error = state[:, 1:]-_state[:, :-1]
print state.shape
plt.plot(error.T)
plt.show()
plt.figure()
for k in range(state.shape[0]):
    plt.plot(state[k, :], 'b-')
    plt.plot(_state[k, :], 'r--')
plt.show()


P = la.solve_discrete_are(A, B, np.eye(A.shape[0]), np.eye(A.shape[0]))
K = -np.dot(la.pinv(np.dot(B.T, np.dot(P, B))+np.eye(A.shape[0])), np.dot(B.T, np.dot(P, A)))

data = {}
data['A'] = A
data['B'] = B
data['P'] = P
data['K'] = K

np.savez(os.path.join(directory, 'feedback'), **data)
print K

#plt.show()


