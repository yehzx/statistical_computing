import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

"""
1. Find two importance function f1 and f2 that are supported on (1, ∞)
and are ”close” to g(x) = x**2 / sqrt(2*pi) * exp(-x**2 / 2), x > 1
Which of your two importance functions should produce the smaller
variance in estimating integration from 1 to inf of g(x) by importance
sampling? Explain.
"""

# set seed = 1000
np.random.seed(1000)

n = 10000

def g(x):
    assert np.all(x > 1), "Range Error of x"
    return x**2 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)


def f1(x):
    # normal distribution pdf
    # but negative part truncated 
    # so the pdf is twice the value of the original one
    return 2 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)


def f2(x):
    # exponential distribution pdf (lambda = 1)
    return np.exp(-x)


def cal_f1_result(n):
    x = np.random.normal(size=n)
    x = np.abs(x) +1
    density = g(x) / f1(x-1)
    theta_hat = np.mean(density)
    convergence_estimation = np.divide(np.cumsum(density), np.arange(1, n+1))
    variance = np.var(density)

    return theta_hat, convergence_estimation, variance


def cal_f2_result(n):
    x = np.random.exponential(size=n) + 1
    density = g(x) / f2(x-1)
    theta_hat = np.mean(density)
    convergence_estimation = np.divide(np.cumsum(density), np.arange(1, n+1))
    variance = np.var(density)

    return theta_hat, convergence_estimation, variance
    

x = np.linspace(1.0001, 10, 100)
plt.rcParams.update({'font.size': 14})

plt.plot(x, g(x), "b", label="g(x)")
plt.plot(x, f1(x), "r", label="f1(x): normal pdf")
plt.plot(x, f2(x), "g", label="f2(x): exp pdf")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Functions")
plt.legend()
plt.show()

f1_theta_hat, f1_con_esti, f1_variance = cal_f1_result(n)
f2_theta_hat, f2_con_esti, f2_variance = cal_f2_result(n)
plt.clf()
plt.plot(np.arange(1, n+1), f1_con_esti, color="r", label="f1(x)")
plt.plot(np.arange(1, n+1), f2_con_esti, color="g", label="f2(x)")
print(f1_theta_hat, f2_theta_hat)
plt.xlabel("Iteration")
plt.ylabel("Integral Estimate")
plt.title("Estimation of Integral")
plt.legend()
plt.show()
print(f1_variance, f2_variance)

