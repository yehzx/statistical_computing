import numpy as np
import matplotlib.pyplot as plt

"""
3. The continuous random variable X with positive support is said to have
the Pareto distribution if its probability density function is given by
f (x) = beta * (alpha**beta) / ((x+alpha) ** (beta+1)), x > 0
Generate a random sample of size 1000 from the Pareto distribution
with alpha = 2 and beta = 4 using ”inverse transformation method”. Compare
the empirical and theoretical distributions by graphing the histogram
of the sample and superimposing the Pareto density curve.
"""

# set seed
np.random.seed(1000)

# # of samples
n = 1000

# Pareto (a, b)
a = 2
b = 4


def pareto(a, b, x):
    assert np.all(x > 0), "Error!"
    return b * (a**b) / ((x+a) ** (b+1))


def inverse_function(a, b, u):
    return ((1-u) / (a**b)) ** (-1/b) - a


def inverse_transformation(a, b, n):
    u = np.random.uniform(0, 1, n)

    return inverse_function(a, b, u)


x = np.linspace(0.01, 10, 100)
density = pareto(a, b, x)

# generate beta variables
pareto_variables = inverse_transformation(a, b, n)

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 14})
plt.xlabel("X")
plt.ylabel("Density")
plt.title("Histogram of Pareto Distribution (alpha=2, beta=4)")
plt.plot(x, density, "b", label="pareto pdf")
plt.xlim(0, 10)
plt.hist(pareto_variables, density=True, bins=30, alpha=0.5)
plt.show()

