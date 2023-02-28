import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


"""
1. Generate a random sample of size 1000 from the Beta(3,2) distribu-
tion using the acceptance-rejection method. Graph the histogram of
the sample with the theoretical Beta(3,2) density superimposed as the
Figure below.
"""

# set seed
np.random.seed(1000)

# # of samples
n = 1000

# Beta (a, b)
a = 3
b = 2

def beta_acceptance_rejection(a, b, n):
    result = []
    
    # first derivative to get maximum density of beta function
    c = st.beta.pdf((1 - a) / (2 - (a+b)), a, b)

    while len(result) < n:
        # I use uniform distribution to cover beta distribution, though there might
        # be some better choices
        uniform_var = np.random.uniform(0, 1)
        
        # calculate the corresponding beta distribution density
        beta_var_density = st.beta.pdf(uniform_var, a, b)
        
        accept_threshold = np.random.uniform(0, 1)

        ratio = beta_var_density / c
        assert ratio <= 1, "Error!"
        
        if ratio > accept_threshold:
            result.append(uniform_var)
    
    return result
    

# theorectical distribution
x = np.linspace(0, 1, 100)
density = st.beta.pdf(x, a, b)

# generate beta variables
beta_variables = beta_acceptance_rejection(a, b, n)

plt.figure(figsize=(8,8))
plt.rcParams.update({'font.size': 14})
plt.plot(x, density, "b", label="beta pdf")
plt.xlim(0, 1)
plt.ylim(bottom=0)
plt.xlabel("X")
plt.ylabel("Density")
plt.title("Histogram of Beta(3,2)")
plt.hist(beta_variables, density=True, bins=30, alpha=0.5)
plt.show()
