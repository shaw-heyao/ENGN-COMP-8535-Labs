import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
from source import GMM_EM, E_step, M_step, incomplete_likelihood, k_means, k_means_pp

########################################################
## This file is for you to test your functions in source.py.
## Do not put test codes directly in source.py.
## You may edit this file however you want.
## This file will be ignored during marking.
########################################################

### Some example tests are provided below for your convenience ###

def plt_gmm(x, mu, Sigma, pi, path):
    '''
    plots the 3 bivariate Gaussians and save to file
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
     - path:  string
              path to save figure
    '''
    assert x.shape[1] == 2, "must be bi-variate Gaussians"
    assert mu.shape[0] == 3, "must have 3 components"
    gamma = E_step(x, mu, Sigma, pi)
    eigval, eigvec = np.linalg.eig(Sigma)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.scatter(x[:,0], x[:,1], 0.01, c=gamma)
    for centroid,alpha in zip(mu,pi):
        for i in range(0,6,2):
            ell = Ellipse(centroid, np.sqrt(eigval[0])*i, np.sqrt(eigval[1])*i,
                          np.rad2deg(np.arctan2(*eigvec[:,0][::-1])), edgecolor='black', fc='None', lw=2, alpha=alpha)
            ax.add_artist(ell)
    plt.grid(linestyle='--', alpha=0.3)
    plt.savefig(path, dpi=300)
    plt.close(fig)

# set random seed for PRNG
np.random.seed(int(time.time()))

# load data
x = np.load('x.npy')

### k-means initialisation ######################
which_component, init_mu = k_means(x, k_means_pp(x, k=3), n_iterations=20)
init_Sigma = np.cov(x - init_mu[which_component], rowvar=False)
_, n_counts = np.unique(which_component, return_counts=True)
init_pi = n_counts / x.shape[0]

mu, Sigma, pi = GMM_EM(x, init_mu, init_Sigma, init_pi)
print(mu)
print(Sigma)
print(pi)
# plot figure and save it as 'result.png'
plt_gmm(x, mu, Sigma, pi, 'result.png')


####### Write your own tests below #######
    