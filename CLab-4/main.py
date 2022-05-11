import numpy as np
import scipy
import scipy.linalg
import time
from source import hard_sparsity_bf, hard_sparsity_omp, hard_equality_lp

########################################################
## This file is for you to test your functions in source.py.
## Do not put test codes directly in source.py.
## You may edit this file however you want.
## This file will be ignored during marking.
########################################################

### Some example tests are provided below for your convenience ###

def sparsity(x, eps):
    '''
    returns no. nonzeros in x
    '''
    return (np.abs(x) > eps).sum()

m, n = 30, 100 # data dimension and no. atoms
s = 3 # at most s entries of X are non-zero
snr = 20 # signal to noise ratio in decibels, greater value means less noise
eps = 1e-4

np.random.seed(int(time.time()))

X = np.zeros(n)
X[np.random.choice(n,s)] = np.random.randn(s) # allow at most s non-zero elements
A = np.random.randn(m,n)
A = A / np.linalg.norm(A, axis=-1, keepdims=True)
A = scipy.linalg.orth(A.T).T # let's make rows orthonormal to ease problem, you may also comment this line

y = A @ X
y = y + np.random.randn(*y.shape) * np.power(10,-0.05*snr) # add some noise to y, you may also comment this line to test hard_equality

# test hard_sparsity
time_start = time.time()
x = hard_sparsity_bf(A, y, s) # you should test hard_sparsity_omp as well
time_end = time.time()

assert sparsity(x, eps) <= s

# should only take a few seconds at most
print(f'||y-Ax||_2={np.linalg.norm(y-A@x)}. Took {time_end-time_start} seconds.')

# test hard_equality
time_start = time.time()
x = hard_equality_lp(A, y)
time_end = time.time()

assert np.abs(A@x-y).max() <= eps

# should only take a few seconds at most
print(f'x is {sparsity(x, eps)}-sparse. Took {time_end-time_start} seconds.')