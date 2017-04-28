import time
from math import sqrt
import numpy as np
from scipy import linalg


def fista(A, b, l, maxit):
    x = np.zeros(A.shape[1])
    pobj = []
    t = 1
    z = x.copy()
    z = np.asmatrix(z).T
    L = linalg.norm(A) ** 2
    time0 = time.time()
    for _ in xrange(maxit):
        xold = x.copy()
        xold = np.asmatrix(xold).T
        print(A.shape,b.shape,z.shape)
        z = z + A.T.dot(b - A.dot(z)) / L
        x = soft_thresh(z, l / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        print (x - np.asmatrix(xold).T).shape
        z = x + ((t0 - 1.) / t)*(x - np.asmatrix(xold).T)
        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
        pobj.append((time.time() - time0, this_pobj))

    times, pobj = map(np.array, zip(*pobj))
    return x

def soft_thresh(x, l):
    print(np.sign(x).shape,np.maximum(np.abs(x) - l, 0.).shape)
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - l, 0.))