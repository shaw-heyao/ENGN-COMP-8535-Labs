import numpy as np
from source import low_rank_approx, constrained_LLS

########################################################
## This file is for you to test your functions in source.py.
## Do not put test codes directly in source.py.
## You may edit this file however you want.
## This file will be ignored during marking.
########################################################

### Some example tests are provided below for your convenience ###

def low_rank_approx(A, k):

    u.s.vt=np,linalg.svd(A)
    u_new=u[:,:k]
    s_new=np.diag(s[:k])
    vt_new=vt[:k]
    X=u_new@s_new@vt_new
    return X

def constrained_LLS(A, B):
    esp=0.0000001
    u,s,vt=np.linalg.svd(B)
    s+=esp
    s1=np.diag(s)
    temp_matrix=sl@vt
    new_matrix=A@temp_matrix.T
    u2,s2,vt2=np.linalg.svd(new_matrix)
    x=vt@s1@vt2[-1]
    return x
    
eps = 0.0001

I3 = np.eye(3)
r2_I3 = low_rank_approx(I3, 2)

assert np.linalg.matrix_rank(r2_I3) <= 2 and np.linalg.norm(I3-r2_I3, ord='fro') <= 1 + eps

D123 = np.diag([1.0, 2.0, 3.0])
x = constrained_LLS(D123, I3)

assert np.abs(np.linalg.norm(x, ord=2) - 1) <= eps and np.linalg.norm(D123@x, ord=2) <= 1 + eps