import numpy as np
import matplotlib.pyplot as plt
"""
2. Generate a random sample of size 1000 from the pdf
f (x) = 2 / pi * (R ** 2) * sqrt(R**2 - x**2),  -R ≤ x ≤ R
using the acceptance-rejection method. Graph the histogram of the
sample with the theoretical density superimposed.
"""

# set seed
np.random.seed(1000)

# Set r = 1
r = 10
n = 1000


def target_function(x, r):
    assert np.all(x <= r) and np.all(x >= -r), f"x not within range!"
    return 2 / (np.pi * (r**2)) * np.sqrt(r**2 - x**2)


def target_acceptance_rejection(r, n):
    result = []
    
    # first derivative to get maximum density of the target function
    # at maximum density x = 0, so c is:
    c = 2 / np.pi / r

    while len(result) < n:
        # Also use uniform distribution to cover the target distribution
        uniform_var = np.random.uniform(-r, r)
        
        # calculate the corresponding target distribution density
        target_var_density = target_function(uniform_var, r)
        
        accept_threshold = np.random.uniform(0, 1)

        ratio = target_var_density / c
        assert ratio <= 1, "Error!"
        
        if ratio > accept_threshold:
            result.append(uniform_var)
    
    return result

x = np.linspace(-r, r, 100)
density = target_function(x, r)

target_variables = target_acceptance_rejection(r, n)

plt.rcParams.update({'font.size': 14})
plt.plot(x, density, "b")
plt.hist(target_variables, density=True, bins=30, alpha=0.5)
plt.xlabel("X")
plt.ylabel("Density")
plt.title("Histogram of the random samples")
plt.show()

