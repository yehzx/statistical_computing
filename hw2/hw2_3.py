import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

"""
3.  Estimate the integral of e ** (x**2) from 0 to 1
using Monte Carlo method with n = 10000 for the following estimators
"""

# set seed = 1000
np.random.seed(1000)

n = 10000


def h(x):
    return np.exp(x ** 2)


def regular_monte_carlo(uni_var):
    var = h(uni_var)
    theta = np.mean(var)
    variance = np.var(var)
    return theta, variance


def antithetic_method(uni_var):
    antithetic_uni = 1 - uni_var
    var1 = h(uni_var)
    var2 = h(antithetic_uni)
    var = (var1 + var2) / 2
    theta = np.mean(var)
    variance = np.var(var)
    return theta, variance


def control_variate_method(uni_var):
    var = h(uni_var)
    control_var = uni_var
    cov_matrix = np.cov(var, control_var, ddof=0)
    c_star = - cov_matrix[1, 0] / cov_matrix[1, 1]
    control_var_adjusted = var + c_star * (control_var - 0.5)
    theta = np.mean(control_var_adjusted)
    variance = np.var(control_var_adjusted)
    return theta, variance


def combined_method(uni_var):
    antithetic_uni = 1 - uni_var
    var1 = h(uni_var)
    var2 = h(antithetic_uni)
    control_var = uni_var
    cov_1_2 = np.cov(var1, var2, ddof=0)
    cov_1_3 = np.cov(var1, control_var, ddof=0)
    cov_2_3 = np.cov(var2, control_var, ddof=0)
    variance_1 = cov_1_2[0, 0]
    variance_2 = cov_1_2[1, 1]
    variance_3 = cov_1_3[1, 1]
    cov_1_2 = cov_1_2[0, 1]
    cov_2_3 = cov_2_3[0, 1]
    cov_1_3 = cov_1_3[0, 1]
    
    n1 = cov_1_3-cov_2_3
    n2 = (variance_1 + variance_2 - cov_1_2)
    n3 = -cov_1_2+variance_2
    c = (-n1*n3 - n2*cov_2_3) / (n2*variance_3 - n1*n1)
    a = (-c*variance_3 - cov_2_3) / (cov_1_3 - cov_2_3)
    combined_estimate = a * var1 + (1-a) * var2 + c * (control_var-1/2)
    theta = np.mean(combined_estimate)
    variance = np.var(combined_estimate)
    return theta, variance
    
uni_var = np.random.uniform(size=n)

print("Regular Monte Carlo:")
theta, variance = regular_monte_carlo(uni_var)
print(f"theta = {theta}, variance = {variance}")
print("Control Variate Method:")
theta, variance = control_variate_method(uni_var)
print(f"theta = {theta}, variance = {variance}")
print("Antithetic Method:")
theta, variance = antithetic_method(uni_var)
print(f"theta = {theta}, variance = {variance}")
print("Combined Method:")
theta, variance = combined_method(uni_var)
print(f"theta = {theta}, variance = {variance}")