import numpy as np
import pylab as pl
import math 

lpc = np.array([[1,-0.1], [1,0.1], [1, -2.0/3], [1, 2.0/3]])

plot_count = len(lpc)

sigma2 = np.arange(0.01, 100, 0.01)
print len(lpc)
x = np.zeros((len(lpc), len(sigma2)))
j = 0
R = [[ 7.20000002,  4.80000002], [ 4.80000002,  7.20000002]]#[-0.66]
R = [[ 7.20000002, -4.80000002], [-4.80000002,  7.20000002]]#[0.66]

#R = [[ 4.04040404,  0.4040404 ], [ 0.4040404,  4.04040404]]# [-0.1]
#R = [[ 4.04040404, -0.4040404 ], [-0.4040404,   4.04040404]]# [0.1]
#R = [[ 4.08333333, 0.58333333], [ 0.58333333,  4.08333333]]# [1/7]
#R = [[ 4.08333333, -0.58333333], [-0.58333333,  4.08333333]]# [-1/7]

#R = [[4.00040004, -0.040004],[-0.040004, 4.00040004]]
#R = [[7.2, -4.8], [-4.8,  7.2]]
#R = [[ 4.08333333, -0.58333334],[-0.58333334, 4.08333333]]

for j in range(len(lpc)):
    print np.dot(lpc[j], np.dot(R, lpc[j]))
    for s in range(len(sigma2)):
        x[j,s] = (2/(np.sqrt(2*math.pi*sigma2[s])))*math.exp(-np.dot(np.dot(lpc[j,:], R), lpc[j,:])/(2*sigma2[s]))

    print sigma2[np.argmax(x[j])]

fig, ax = pl.subplots()

#pl.hold(True)
for j in range(0, len(lpc)):
    ax.plot(sigma2, x[j], label=str(j))
ax.legend(loc='upper right')

pl.show()
