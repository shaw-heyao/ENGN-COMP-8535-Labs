# Coding Lab 2: MDS and Statistical Shape Analysis

## Overview

In this Coding lab, you will be asked to code the classical Multidimensional Scaling technique and perform shape analysis with NumPy. Use workflow described below for submitting your codes:
1. Fork this repository and clone your fork to your computer.
2. Read the requirements and complete source script(s). You should define your functions and run the main function in `classical_mds.py`.
3. Commit and push.

Your coding lab **will not be marked**. It is for your practice only.

## Questions

### 1. Classical MDS (difficulty: :star: :star: :star:)

You have be given 12 Australian cities’ world coordinates (in latitude, longitude). Let us for now assume the Australian continent is purely flat (i.e. all cities are on the sea level). The 12 cities are roughly covering uniformly the entire Australia continent.

Compute the pairwise Euclidean distances between every pair of cities, add some tiny amount of Gaussian noise to the distances, and build a distance matrix D of size 12×12. Implement the classical MDS algorithm using Python/Numpy, which takes a 12×12 distance matrix D as input, and returns the coordinates of the 12 cities as output.

The steps in your code are:
1. Load the input distance matrix
2. Run MDS

### 2. Procrustes Analysis (difficulty: :star: :star: :star: :star:)

Following the previous example, align (overlay) the computed 12 cities with a map of Australia, visually compare your estimated city locations with the ground-truth city locations. (To achieve the correct alignment, you may need to rotate, flip, translate, and re-scale your computed coordinates.)

The steps in your code are:
1. Align the computed locations so that they match the actual map
2. Plot the map, and overlay with your aligned cities, with the city name next to each city

HINT: To perform this alignment, you need to perform an orthogonal procrustes analysis (check the link in the code for more details).The simplest algebraic statement of a Procrustes problem seeks a matrix <img src="https://render.githubusercontent.com/render/math?math=T">, that minimises,

<img src="https://render.githubusercontent.com/render/math?math=Q^{opt}=\arg_{\{Q|Q^{-1}=Q^T\}}\min ||X_1Q - X_2||^2_F">

over <img src="https://render.githubusercontent.com/render/math?math=Q\in R^{p_1xp_2}">, for given <img src="https://render.githubusercontent.com/render/math?math=X_1\in R^{nxp_1}">, <img src="https://render.githubusercontent.com/render/math?math=X_2\in R^{nxp_2}">, where in our case <img src="https://render.githubusercontent.com/render/math?math=T"> is the orthogonal matrix <img src="https://render.githubusercontent.com/render/math?math=Q">. <img src="https://render.githubusercontent.com/render/math?math=X_1"> is the matrix to be matched into the reference matrix <img src="https://render.githubusercontent.com/render/math?math=X_2">.

<img src="https://render.githubusercontent.com/render/math?math=Q"> refers to the matrix's rotation. To perform scaling and rotation, we need to solve a more elaborate problem,

<img src="https://render.githubusercontent.com/render/math?math=(Q^{opt}, s^{opt}) =\arg_{\{(Q, s)|Q^{-1}=Q^T\}}\min||sX_1Q - X_2||^2_F">

and finally, the translation could be expressed as follows,

<img src="https://render.githubusercontent.com/render/math?math=t = \mu_{X_2} - s^{opt} \mu_{X_1}Q^{opt}">

Both matrices <img src="https://render.githubusercontent.com/render/math?math=X_1, X_2"> should be zero-centered and with unit norms. This should work as a pre-scaling step.

The final transformation <img src="https://render.githubusercontent.com/render/math?math=Z"> should be

<img src="https://render.githubusercontent.com/render/math?math=Z = sX_1 Q + t">

### 3. Test your codes (difficulty: :star:)

Run the `main` function to test the code. Verify that the distance matrix is correct using your intuition. Also, verify the correct alignment of the cities from the plot.
