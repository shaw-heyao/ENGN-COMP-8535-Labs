import numpy as np
import cvxpy as cp

########################################################
## Complete functions in skeleton codes below
## following instructions in each function.
## Do not modify existing function name or inputs.
## Do not test your codes here - use main.py instead.
## You may use any built-in functions from NumPy.
## You may define and call new functions as you see fit.
########################################################


def hard_sparsity_bf(A, y, s):
    '''
    inputs: 
      - A: m-by-n matrix
      - y: m-dimensional vector
      - s: integer in range (0,m)
    returns:
      x: n-dimensional vector that minimises ||y-Ax||_2 subject to s-sparsity
    '''
    return ...

def hard_sparsity_omp(A, y, s):
    '''
    inputs: 
      - A: m-by-n matrix
      - y: m-dimensional vector
      - s: integer in range (0,m)
    returns:
      x: n-dimensional vector that minimises ||y-Ax||_2 subject to s-sparsity
    '''
    return ...

def hard_equality_lp(A, y):
    '''
    inputs: 
      - A: m-by-n matrix
      - y: m-dimensional vector
    returns:
      x: n-dimensional vector that is as sparse as possible subject to y=Ax
         sparsity is approximated by minimising L1 norm ||x||_1 instead
    '''
    # you can use cvxpy to solve the linear programming
    return ...



### you can optionally write your own functions like below ###

# def my_func_name(input1, input2, ...):
#     do something
#     return ...