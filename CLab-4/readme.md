# Coding Lab 4

## Questions

### Requirements

Your `source.py` should be a self-contained file that includes only function definitions. 
* It should not invoke/import functions from other file(s)/package(s) than NumPy, itself or CVXPY.
* Your codes should be reasonably fast. All test cases are expected to finish within a few seconds when we mark your codes. A few tips to maximize efficiency in general:
  - Simplify arithmetic operations by adding your own knowledge of the problem, *e.g.* there are three ways to compute eigenvalue decomposition of a matrix <img src="https://render.githubusercontent.com/render/math?math=B=A^TA">
 where A is an m-by-n matrix with m<<n, 
  ```python
  np.linalg.eig(A.T @ A) # the slowest and returns complex numbers with zero imaginary parts
  np.linalg.eigh(A.T @ A) # faster, exploit the fact that B is symmetric
  np.linalg.svd(A, full_matrices=False) # much faster, exploit the fact B is also low rank
  ```
  - Use tensor-level operations to replace slow loops, *e.g.* instead of

  ```python
  # A is 2D numpy matrix
  C = np.empty_like(A)
  for i in A.shape[0]: # loops are slow
      for j in A.shape[1]: # nested loops are slower
          C[i,j] = A[i,j] * i
  ```

  you can write loop-free codes like
  ```python
  C = np.arange(A.shape[0]).reshape(-1,1) * A
  ```
  - Many optimisation problems are impossible to solve exactly on mordern computers (*e.g.* NP-hard), however a good approximation (suboptimal solution) can often be found efficiently.

### 1. Compressive sensing

Given an m-dimensional vector x and an m-by-n sensing matrix A (m<n), your task is to find an as-sparse-as-possible vector x that satisfies y=Ax as well as possible, *i.e.* an x that contains minimal number of non-zero elements under <img src="https://render.githubusercontent.com/render/math?math=y\approx Ax">.

One could argue the above problem is not well-defined from a mathematical point of view -- there is always a trade-off between how sparse x is and how close Ax and y are. So let us consider two (somewhat extreme) ways to formalise this task.

#### Hard sparsity

A vector x is called *s-sparse* if it has **at most** s nonzero element. Given s, write a function that returns an s-sparse vector x that minimises the distance <img src="https://render.githubusercontent.com/render/math?math=\|Ax-y\|_2">. The problem is NP-hard but the minimimal solution can be approximated, e.g. by orthogonal matching pursuit (OMP).

*I.e.*, solve the optimisation problem

<img src="https://render.githubusercontent.com/render/math?math=\arg_x\min \|Ax-y\|_2\ s.t. \|x\|_0\le s">


##### Brute-force solver (difficulty: :star: :star: :star:)
Since there are at most <img src="https://render.githubusercontent.com/render/math?math=(^n_s)"> nonzero elements in x, a naive solution would be to enumerate all <img src="https://render.githubusercontent.com/render/math?math=(^n_s)"> possible subsets of nonzero elements. For each subset we may simply extract the correponding columns in A and solve a least square problem of s variables. Write a function `hard_sparsity_bf(A,y,s)` that loops through all <img src="https://render.githubusercontent.com/render/math?math=(^n_s)"> combinations of s nonzero coefficients, and returns the minimal s-sparse solution x. Consider how running time would change with increasing n and s.

Hint: you can use below function to generate all <img src="https://render.githubusercontent.com/render/math?math=(^n_s)"> combinations of indices.
```python
def combs_idx(n, k):
    '''
    returns an index array of n chooses k.
    arguments:
        -n: integer
        -k: integer
    returns:
        np array of shape (C,k)
        where C is number of combinations.
        each row has k integers in [0,n), representing indices of a k-combination.
    '''
    assert n>=k
    combs = []
    comb = np.arange(k)
    while comb[-1] < n: 
        combs.append(comb.copy())
        for i in range(1,k+1):
            if comb[-i] != n-i:
                break # find last occurance of non-maximum elem
        # reset this last part in increasing order
        comb[-i:] = np.arange(1+comb[-i], i+1+comb[-i])
    return np.array(combs)
```

##### Orthogonal Matching Pursuit (difficulty: :star: :star: :star:)
Now implement orthogonal matching pursuit in function `hard_sparsity_omp(A,y,s)`. Consider what advantages and disadvantages OMP has compared to the brute-force solver.

#### Hard equality (difficulty: :star: :star:)

Now consider the other extreme where the equality y=Ax must hold. Complete the function `hard_equality_lp(A,y)` so it returns a vector x that strictly satisfies <img src="https://render.githubusercontent.com/render/math?math=Ax-y=\mathbf{0}"> but has as few nonzero elements as possible. 

Again this problem is NP-hard, however an approximated solution can be found if we replace the L0 norm with L1 norm, in which case the problem becomes convex can be globally minimised by any standard linear programming solver.

*I.e.*, solve the approximated optimisation problem
<img src="https://render.githubusercontent.com/render/math?math=\arg_x\min \|x\|_1\,s.t. Ax-y=\mathbf{0}">


Consider how this modified problem differs from the L0 objective, and why it still leads to a sparse solution.

Hint: 
* Use CVXPY package to efficiently solve a [linear programming](https://www.cvxpy.org/examples/basic/linear_program.html). You can install CVXPY by 
```
python3 -m pip install cvxpy
```

### 2. Test your codes (difficulty: :star: :star:)

A testing framework is provided in `main.py` for your convenience. You should modify it or write your own tests, however they will not be reflected in your marks.
