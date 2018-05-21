
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib2tikz import save as tikz_save
import scipy.signal as signal

import numpy.random as rand
import scipy.linalg as la

import primitive.SubspacePrim as subspace

def brownian_motion(length, dims, sigma, dt):
    out = rand.normal(loc=0., scale=sigma*dt, size=(length, dims))
    out = np.cumsum(out, axis=0)

    return out


def enforce_full_rank(X, epsilon=0.1):
    # enforce a minimum rank of margin epsilon
    U,s,V = la.svd(X, full_matrices=False)
    s[s<epsilon] = epsilon

    return np.dot(U, np.dot(np.diag(s), V))
    
def stabilize_operator(X, epsilon=0.1):
    # enforce a minimum rank of margin epsilon
    U,s,V = la.svd(X, full_matrices=False)

    s[s<epsilon] = epsilon
    s = 0.99*s/s[0]

    return np.dot(U, np.dot(np.diag(s), V))


# generate random discrete system matrices
n = 5  # number of states
m = 4  # number of inputs
k = 2  # number of output variables

samples = 1000000
history = 10
future = 1#50
past = history-future

internal_dim = 5


# stable system matrix 
#A = enforce_full_rank((rand.random((n, n))-0.5)*2.)
#while la.norm(A, 2) > 1:
    #print '.'
    #A = A/la.norm(A, 2)
    #A = enforce_full_rank(A)
A = stabilize_operator((rand.random((n, n))-0.5)*2.)


B = enforce_full_rank((rand.random((n, m))-0.5)*2.)
C = enforce_full_rank((rand.random((k, n))-0.5)*2.)
D = enforce_full_rank((rand.random((k, m))-0.5)*2.)
#C = enforce_full_rank(rand.random((k, n)))
#D = enforce_full_rank(rand.random((k, m)))

X = np.zeros((samples, n))
Y = np.zeros((samples, k))

U = brownian_motion(samples, m, sigma=5., dt=1./samples) 

plt.figure()
for _m in range(m):
    plt.plot(U[:, _m])
    
plt.title('Input sample paths')
plt.show()

for t in range(samples-1):
    X[t+1] = np.dot(A, X[t]) + np.dot(B, U[t])
    Y[t] = np.dot(C, X[t])

Y[samples-1] = np.dot(C, X[samples-1])

plt.figure()
for _n in range(n):
    plt.plot(X[:, _n])
    
plt.title('State sample paths')

plt.figure()
for _k in range(k):
    plt.plot(Y[:, _k])
    
plt.title('Output sample paths')

#plt.show()

ss = subspace.PrimLearn()
obs = np.append(Y, U, axis=1)
#ss.LoadRawData(np.append(Y, U, axis=1).T)
ss.LoadRawData(obs.T)
ss.PreprocessData(past=past, future=future, overlap=False)
ss.SubspaceDFA(internal_dim)


h = np.zeros((X.shape[0], internal_dim))
Y_hat = np.zeros(Y.shape)
for t in range(past, samples):
    _Xp = np.reshape(obs[t-past:t, :], (-1, 1))
    h[t] = np.dot(ss.K, _Xp).flatten()

    _Xf = np.dot(ss.O, h[t])
    Y_hat[t] = _Xf[:k]


plt.figure()
for _k in range(k):
    plt.plot(Y[:, _k])
    plt.plot(Y_hat[:, _k], '--')
plt.title('Output trackings')

plt.figure()
for _n in range(internal_dim):
    plt.plot(h[:, _n])
    
plt.title('Inferred sample paths')
plt.show()

phi = np.dot(h.T, la.pinv(X.T))
print phi
