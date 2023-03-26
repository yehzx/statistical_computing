import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

"""
2. We want to compute the following integral by Monte Carlo:
integral of exp(-x*cos(pi*x)) from 0 to 1 = E[h(U)],
By comparing a Monte Carlo estimator with and without control
variate, please find the variance reduction from the use of 
control variate.

"""

# set seed = 1000
np.random.seed(1000)

n = 10000


def h(x):
    return np.exp(-x * np.cos(np.pi*x))


def regular_monte_carlo(uni_var):
    var = h(uni_var)
    theta = np.mean(var)
    variance = np.var(var)
    return theta, variance


def control_variate_method(uni_var):
    var = h(uni_var)
    control_var = np.exp(-uni_var)
    mean_var = np.mean(control_var)
    cov_matrix = np.cov(var, control_var, ddof=0)
    c_star = - cov_matrix[1, 0] / cov_matrix[1, 1]
    control_var_adjusted = var + c_star * (control_var - mean_var)
    theta = np.mean(control_var_adjusted)
    variance = np.var(control_var_adjusted)
    return theta, variance


uni_var = np.random.uniform(size=n)
simple_theta, simple_variance = regular_monte_carlo(uni_var)
control_theta, control_variance = control_variate_method(uni_var)
variance_reduction = (simple_variance - control_variance) / simple_variance

print(simple_theta, control_theta)
print(simple_variance, control_variance)
print(variance_reduction)
