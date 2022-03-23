import numpy as np

########################################################
# Complete functions in skeleton codes below
# following instructions in each function.
# Do not modify existing function name or inputs.
# Do not test your codes here - use main.py instead.
# You may use any built-in functions from NumPy.
# You may define and call new functions as you see fit.
########################################################


def low_rank_approx(A, k):
    '''
    inputs: 
      - A: m-by-n matrix
      - k: positive integer, k<=m, k<=n
    returns:
      X: m-by-n matrix that is an as-close-as-possible approximation of A
         up to rank k
    '''
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    return u[:, :k] @ np.diag(s[:k]) @ vt[:k]


def constrained_LLS(A, B):
    '''
    inputs:
      - A: n-by-n full rank matrix
      - B: n-by-n matrix
    returns:
      x: n-diemsional vector that minimises ||Ax||2 subject to ||Bx||2=1 
    '''
    eps = 1e-6  # small value to handle singular matrix
    _, s, vt = np.linalg.svd(B)
    s += min(eps, eps*s[0])  # add small value in case rank deficiency
    W = np.diag(s) @ vt  # full rank n-by-n matrix
    x = np.linalg.svd(A @ W.T)[2][-1]  # the smallest right singular vector
    return vt.T @ np.diag(1/s) @ x


### you can optionally write your own functions like below ###

# def my_func_name(input1, input2, ...):
#     do something
#     return ...
