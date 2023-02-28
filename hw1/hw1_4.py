import numpy as np
import scipy.stats as st
from scipy.linalg import cholesky
import time

"""
4. Please compare the efficiencies of two methods for generating the Wishart
Distribution with n = 100 and covariance sigma = [[1, 0.5], [0.5, 2]]
"""

# set seed
np.random.seed(1000)

iter = 1000
n = 100
d = 2

cov = [[1, 0.5], [0.5, 2]]


def timer(iter):
    def wrap(func):
        def inner_func(*args, **kwargs):
            start = time.time()
            for i in range(iter):
                func(*args, **kwargs)
            end = time.time()
            print(f"Elapsed time: {round(end - start, 3)}sec")
        return inner_func
    return wrap


def build_matrix_A(n, d):
    A = np.zeros(shape=(d, d))
    for i in range(1, d):
        for j in range(0, i):
            A[i, j] = np.random.normal()
    for i in range(0, d):
        A[i, i] = np.sqrt(np.random.chisquare(df=n-i+1))
    
    return A

@timer(iter)
def generate_wishart_bartlett(n, cov):
    d= np.ndim(cov)
    A = build_matrix_A(n, d)
    L = cholesky(cov)
    M = L @ A @ A.T @ L.T

    return M

@timer(iter)
def generate_wishart_builtin(n, cov):
    return st.wishart.rvs(n, cov)

@timer(iter)
def generate_wishart_original(n, cov):
    X = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=n)
    M = X.T @ X

    return M

print(f"n = {n}, iteration = {iter}")
print("Generate Wishart distribution by Bartlett’s decomposition")
generate_wishart_bartlett(n, cov)
print("Generate Wishart distribution by generating n*d data matrix")
generate_wishart_original(n, cov)
print("Generate Wishart distribution from scipy built-in method")
generate_wishart_builtin(n, cov)