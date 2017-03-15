import numpy as np
import numpy.linalg as ln

def SubspaceDFA(Xp, Xf, k):
    """Decompose linear prediction matrix into O and K matrices"""
    # compute predictor matrix
    F = np.dot(Xf, ln.pinv(Xp))
    #gamma_f = ln.cholesky(np.cov(Xf))
    #gamma_p = ln.cholesky(np.cov(Xp))

    [U, S, Vh] = ln.svd(F)

    U = U[:, 0:k]
    S = np.diag(S[0:k])
    Vh = Vh[0:k, :]

    #K = np.dot(np.dot(np.sqrt(S), Vh), ln.pinv(gamma_p))
    K = np.dot(np.sqrt(S), Vh)

    #O = np.dot(np.dot(gamma_f, U), np.sqrt(S))
    O = np.dot(U, np.sqrt(S))

    return [O, K]


if __name__ == "__main__":
    print "Subspace DFA Test"

    import numpy.random as rand

    u = np.cumsum(rand.rand(1, 100000)*0.1, axis=1);
    x = np.array([np.cos(u), np.sin(u)])

    import pylab as pl
    import matplotlib.cm as cm
    #pl.scatter(x[1, :], x[0,:])
    #pl.show()

    f = 10
    p = 10
    l = f+p
    d = x.shape[0]
    Xl = x.T.reshape(-1, l*d).T # reshape into column vectors of length 20
    print Xl.shape
    Xf = Xl[(p*d):(l*d), :]
    Xp = Xl[0:(p*d), :]
    print Xf.shape
    print d, p*d, l*d

    [O, K] = SubspaceDFA(Xp, Xf, 2)
    h = np.dot(K, Xp)
    #pl.plot(h[0,:])
    #pl.show()
    x_hat = np.dot(O, h)

    _x_hat_ = x_hat.T.reshape(-1, d).T
    _Xf_ = Xf.T.reshape(-1, d).T
    err = np.abs(_x_hat_-_Xf_)
    print err.shape
    pl.plot(err[0, :])
    pl.hold(True)
    pl.plot(err[1, :])
    pl.hold(False)
    pl.show()
    pl.figure()
    pl.hold(True)
    #print _Xf_.T.shape
    import itertools
    colors = iter(cm.rainbow(np.linspace(0, 1, f)))
    for n in range(10):
        pl.scatter(_x_hat_[0, n:-1:10], _x_hat_[1, n:-1:10], color=next(colors))
    #pl.hold(True)
    #pl.scatter(_Xf_[0, :], _Xf_[1,:])
    pl.show()
    #print _Xf_.T.shape
