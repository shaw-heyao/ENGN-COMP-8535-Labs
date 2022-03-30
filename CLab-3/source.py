import numpy as np

########################################################
## Complete functions in skeleton codes below
## following instructions in each function.
## Do not modify existing function name or inputs.
## Do not test your codes here - use main.py instead.
## You may use any built-in functions from NumPy.
## You may define and call new functions as you see fit.
########################################################


# This function is already implemented for you.
# Please read the codes here but do not modify this function.
def k_means(X, centroids, n_iterations):
    '''
    standard k-means algorithm
    arguments:
     - X:          np.ndarray of shape [no_data, no_dimensions]
                   input data points
     - centroids:  np.ndarray of shape [k, no_dimensions]
                   centres of initial custers
     - n_iterations: integer, number of iterations to run k-means for
    returns:
     - which_component: np.ndarray of shape [no_data] and integer data
                        type, contains values in [0, k-1] indicating which
                        cluster each data point belongs to
     - centroids:  np.ndarray of shape [k, no_dimensions], centres of 
                   final custers, ordered in such way as indexed by
                   `which_component`
    '''
    k = centroids.shape[0]
    for _ in range(n_iterations):
        # reassign data points to components
        distances = np.linalg.norm(np.expand_dims(X, axis=1) - centroids, axis=-1, ord=2)
        
        which_component = np.argmin(distances, axis=-1)
        # calcuate centroid for each component
        centroids = np.stack(list( X[which_component==i].mean(axis=0) for i in range(k) ), axis=0)

    return which_component, centroids

# This function is already implemented for you.
# Please read the codes here but do not modify this function.
def GMM_EM(x, init_mu, init_Sigma, init_pi, epsilon=0.001, maxiter=100):
    '''
    GMM-EM algorithm with shared covariance matrix
    arguments:
     - x:          np.ndarray of shape [no_data, no_dimensions]
                   input 2-d data points
     - init_mu:    np.ndarray of shape [no_components, no_dimensions]
                   means of Gaussians
     - init_Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
                   covariance matrix of Gaussians
     - init_pi:    np.ndarray of shape [no_components]
                   prior probabilities of P(z)
     - epsilon:    floating-point scalar
                   stop iterations if log-likelihood increase is smaller than epsilon
     - maxiter:    integer scaler
                   max number of iterations
    returns:
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    '''
    mu = init_mu
    Sigma = init_Sigma
    pi = init_pi
    no_iterations = 0
    
    # compute log-likelihood of P(x)
    logp = np.log(incomplete_likelihood(x, mu, Sigma, pi)).sum()
    print("Init log P(x) is {:.4e}".format(logp))
    
    while True:
        no_iterations = no_iterations + 1
        
        # E step
        gamma = E_step(x, mu, Sigma, pi)
        # M step
        mu, Sigma, pi = M_step(x, gamma)
        
        # exit loop if log-likelihood increase is smaller than epsilon
        # or iteration number reaches maxiter
        new_logp = np.log(incomplete_likelihood(x, mu, Sigma, pi)).sum()
        if new_logp < epsilon + logp or no_iterations > maxiter:
            print("Iteration {:03} log P(x) is {:.4e}".format(no_iterations, new_logp))
            break 
        else:
            print("Iteration {:03} log P(x) is {:.4e}".format(no_iterations, new_logp), end="\r")
            logp = new_logp

    return mu, Sigma, pi

# This function is already implemented for you.
# Please read the codes here but do not modify this function.
def incomplete_likelihood(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - px:    np.ndarray of shape [no_data]
              probabilities of P(x) at each data point in x
    '''
    p = complete_likelihood(x, mu, Sigma, pi)
    return p.sum(axis=-1)
    

def E_step(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - gamma: np.ndarray of shape [no_data, no_components]
              probabilities of P(z|x)
    '''
    p = complete_likelihood(x, mu, Sigma, pi)
    return p / p.sum(axis=-1,keepdims=True)

def M_step(x, gamma):
    '''
    
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - gamma: np.ndarray of shape [no_data, no_components]
              probabilities of P(z|x)
    returns:
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    '''
    N_k = gamma.sum(axis=0) # [no_components]
    mu = gamma.T @ x / N_k[:,np.newaxis]
    pi = N_k / x.shape[0]
    
    _, no_dimensions = x.shape
    deviation = (np.expand_dims(x,axis=1) - mu).reshape((-1, no_dimensions)) #[no_data*no_comp, no_dim]
    Sigma = deviation.T * gamma.flatten() @ deviation / gamma.sum()
    return mu, Sigma, pi

def incomplete_likelihood(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - px:    np.ndarray of shape [no_data]
              probabilities of P(x) at each data point in x
    '''
    p = complete_likelihood(x, mu, Sigma, pi)
    return p.sum(axis=-1)


def complete_likelihood(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - p:     np.ndarray of shape [no_data, no_components]
              joint probabilities of P(x,z)
    '''
    no_dimensions = x.shape[-1]
    # compute Gauss density
    deviation = (np.expand_dims(x,axis=1) - mu)
    pdf = np.exp((-0.5 * deviation @ np.linalg.pinv(Sigma) * deviation).sum(axis=-1)) /\
        np.power(2*np.pi, 0.5*no_dimensions) / np.sqrt(pdet(Sigma)) 
    return pdf * pi    

### complete functions below this line ###

def k_means_pp(X, k):
    '''
    Compute initial custer for k-means
    arguments:
     - X:          np.ndarray of shape [no_data, no_dimensions]
                   input data points
    returns:
     - centroids:  np.ndarray of shape [k, no_dimensions]
                   centres of initial custers
    '''
    return None    


### you can optionally write your own functions like below ###

def pdet(M):
    '''
    *From gitlab readme of this homework
    returns the pseudo-determinant of a square matrix
    M must be positive semi-definite
    '''
    rank = np.linalg.matrix_rank(M)
    sigmas = np.linalg.svd(M, compute_uv=False) # by default numpy's svd has |U|=|V|=1
    return np.prod(sigmas[0:rank])

# def my_func_name(input1, input2, ...):
#     do something
#     return ...
