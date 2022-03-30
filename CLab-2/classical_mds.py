from tkinter import N
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib as mpl

australian_cities = np.array([[-31.952222, 115.858889],
                              [-20.31, 118.601111],
                              [-28.774444, 114.608889],
                              [-37.813611, 144.963056],
                              [-42.880556, 147.325],
                              [-34.928889, 138.601111],
                              [-27.467778, 153.028056],
                              [-16.930278, 145.770278],
                              [-12.438056, 130.841111],
                              [-33.865, 151.209444],
                              [-23.7, 133.87],
                              [-20.733333, 139.5]])

australian_city_names = np.array(['Perth', 'Port Hedland', 'Geraldton', 'Melbourne', 'Hobart', 'Adelaide', 'Brisbane', 'Cairns', 'Darwin', 'Sydney',
                                  'Alice Springs', 'Mount Isa'])


# Compute the distance matrix using pairwise Euclidean Distance.
def compute_distance_matrix(gaussian_noise):
    distance_matrix = np.sqrt(np.sum(
        np.square(australian_cities[None, :] - australian_cities[:, None]), axis=-1))
    if gaussian_noise:
        row, col = distance_matrix.shape
        distance_matrix += np.random.randn(row, col)

    return distance_matrix


def print_distance_matrix(distance_M):
    table = tabulate(distance_M, australian_city_names, tablefmt='fancy_grid')
    print(table)


def classical_mds(distance_matrix, k):
    '''
    Returns the estimations alongside with the eigenvalues.
    '''
    n = len(distance_matrix)

    # Matrix mean
    H = np.eye(n) - np.ones((n, n)) / n

    # Compute the B matrix
    B = -H @ np.square(distance_matrix) @ H / 2

    # Get the eigenvalues and eigenvectors of B
    evals, evecs = np.linalg.eigh(B)

    # Sort the eigenvalues in descending order
    sorting_index = np.argsort(evals)[::-1]
    evals = evals[sorting_index]
    evecs = evecs[:, sorting_index]

    # Compute the new coords using top k pcs
    w = np.array(range(k))
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]

    Y = V @ L

    return Y, evals


def plot_the_map(estimated_cities):
    COLOR_ORIGINAL = 'red'
    COLOR_ESTIMATED = 'blue'
    fig = plt.figure(figsize=(10, 7))
    australia = plt.imread('Australia_location_map.png')
    for coord, city in enumerate(australian_city_names):
        y = australian_cities[coord][0]
        x = australian_cities[coord][1]
        estimated_y = estimated_cities[coord][0]
        estimated_x = estimated_cities[coord][1]
        original = plt.scatter(x, y, zorder=1, marker='8', color='red')
        mpl.rcParams['text.color'] = COLOR_ORIGINAL
        plt.text(x + 0.3, y + 0.3, city, fontsize=9)

        estimated = plt.scatter(estimated_x, estimated_y,
                                zorder=1, marker='x', color='blue')
        mpl.rcParams['text.color'] = COLOR_ESTIMATED
        plt.text(estimated_x - 1, estimated_y - 1, city, fontsize=9)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend((original, estimated), ('Original Location of the cities', 'Estimated MDS Location of the cities'),
               scatterpoints=1,
               loc='lower left',
               ncol=1,
               fontsize=8
               )
    plt.imshow(australia, zorder=0, extent=[
               113.155 - 1.5, 153.638889 + 1.4, -43.643611 - 0.9, -10.683333 + 1.7])
    plt.title('Overlay Map of Australia', fontdict={'size': 11})
    plt.show()


def procrustes(X, Y):
    """
    Solves the (Q,s) = argmin ||sXQ - Y||.

    The following code is the implementation of the procrustes analysis as described in the following:
    https://en.wikipedia.org/wiki/Procrustes_analysis
    https://au.mathworks.com/help/stats/procrustes.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html#scipy-spatial-procrustes

    Returns the residual error (d), the new transformation (Z), and the transformation values (tform).
    """
    # Pre-scaling step
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)

    X = X - mean_X
    Y = Y - mean_Y

    sum_sq_X = np.sum(np.square(X))
    sum_sq_Y = np.sum(np.square(Y))

    frobenius_norm_X = np.sqrt(sum_sq_X)
    frobenius_norm_Y = np.sqrt(sum_sq_Y)

    X /= frobenius_norm_X
    Y /= frobenius_norm_Y

    # Solve the original orthogonal procrustes problem to find Q
    A = Y.T @ X
    U, S, V_T = np.linalg.svd(A, full_matrices=False)
    V = V_T.T

    # Solution of Q rotation
    Q = V @ U.T

    trace_A = np.sum(S)

    # Solve the scaled orthogonal procrustes problem to find s by keeping Q fixed
    s = trace_A * frobenius_norm_Y / frobenius_norm_X

    # translation based on the given equation
    t = mean_Y - s * mean_X @ Q

    # final transformation
    Z = X @ Q * s * frobenius_norm_X + t

    # residual sum-of-squares
    d = 1 - np.square(trace_A)

    # transformation values
    tform = {'rotation': Q, 'scale': s, 'translation': t}

    return d, Z, tform


def main():
    distance_M = compute_distance_matrix(gaussian_noise=False)
    print_distance_matrix(distance_M)
    estimated_distance_M, _ = classical_mds(distance_M, 2)
    print("Estimated Distances:\n", estimated_distance_M)
    residual, Z, disparity = procrustes(
        estimated_distance_M, australian_cities)
    print("Australian cities:\n", australian_cities)
    print("Residual Error:", residual)
    print("Tranformed Cities:\n", Z)
    print("Disparity data\n", disparity)
    ##
    # Hand-coding alignment
    ##
    # estimated_distance_M.T[0, :] -= 40
    # estimated_distance_M.T[1, :] += 130
    # plot_the_map(estimated_distance_M)
    plot_the_map(Z)


if __name__ == '__main__':
    main()
