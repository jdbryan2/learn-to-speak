
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
    


# generate random discrete system matrices
n = 3  # number of states
m = 2  # number of inputs
k = 2  # number of output variables

samples = 10000
past = 10
future = 10

# stable system matrix 
A = enforce_full_rank(rand.random((n, n)))
while la.norm(A, 2) > 1:
    print '.'
    A = A/la.norm(A, 2)
    A = enforce_full_rank(A)

U,S,V = la.svd(A)
print S

B = enforce_full_rank(rand.random((n, m)))
U,S,V = la.svd(B)
print S


C = enforce_full_rank(rand.random((k, n)))
U,S,V = la.svd(C)
print S


D = enforce_full_rank(rand.random((k, m)))
U,S,V = la.svd(D)
print S

X = np.zeros((samples, n))
#X[0] = 1  # impulse response
Y = np.zeros((samples, k))
#U = rand.random((samples, m))
#U = np.zeros((samples, m))

U = brownian_motion(samples, m, 1., 1./samples) 

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
ss.PreprocessData(past=past, future=future)
ss.SubspaceDFA(3)


h = np.zeros(X.shape)
for t in range(past, samples):
    _Xp = np.reshape(obs[t-past:t, :], (-1, 1))
    h[t] = np.dot(ss.K, _Xp).flatten()


plt.figure()
for _n in range(n):
    plt.plot(h[:, _n])
    
plt.title('Inferred sample paths')
plt.show()

