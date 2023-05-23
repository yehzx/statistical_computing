import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import math

"""
2. Use the Gibbs sampler to generate a chain with target joint density f(x; y).
Use the Gelman-Rubin convergence method to monitor the convergence of the chain.
Repeat the analysis in the Example.
"""

a = 3
b = 7
n = 30
x0 = 10

sample_num = 600
burnin = 100
# convergence is achieved quickly in this example, so set the mininum number for
# averaging across samples to 10 only 
min_num_for_averaging = 10

# generate 5 chains
k = 5

np.random.seed(1000)


def run_gibbs_sampling_and_convergence_test():
    many_results = []

    # discard first 100 samples
    for i in range(k):
        many_results.append(gibbs_sampler()[burnin:])

    # plot only the first chain
    plot_results(many_results[0])

    # calculate gr statistic from at least #min_num_for_averaging samples
    gelman_rubin_statistic_li = np.array([gelman_rubin_method(
        i, many_results) for i in range(min_num_for_averaging, sample_num + 1)])
    plot_gr_method(gelman_rubin_statistic_li)


def gibbs_sampler():
    x = x0
    sample_li = []
    for i in range(sample_num):
        y = distribution_conditioned_on_x(x)
        x = distribution_conditioned_on_y(y)
        sample_li.append((x, y))

    return sample_li


def distribution_conditioned_on_x(x):
    return np.random.beta(x + a, n - x + b)


def distribution_conditioned_on_y(y):
    return np.random.binomial(n, y)


def gelman_rubin_method(first_n_samples, many_results):
    # calculate statisitics from the given first n samples
    many_results = np.array([result[:first_n_samples]
                            for result in many_results])
    chain_mean = np.mean(many_results, axis=1)
    chain_mean_x = chain_mean[:, 0]
    chain_mean_y = chain_mean[:, 1]
    between_chain_var_x = np.var(chain_mean_x) * first_n_samples
    between_chain_var_y = np.var(chain_mean_y) * first_n_samples

    # many_results[#chain, #sample, x_or_y]
    pooled_within_chain_var_x = sum(
        [np.var(many_results[i, :, 0]) for i in range(k)]) / k
    pooled_within_chain_var_y = sum(
        [np.var(many_results[i, :, 1]) for i in range(k)]) / k

    gelman_rubin_statistic_x = \
        ((first_n_samples - 1) * pooled_within_chain_var_x + between_chain_var_x) \
        / first_n_samples / pooled_within_chain_var_x
    gelman_rubin_statistic_y = \
        ((first_n_samples - 1) * pooled_within_chain_var_y + between_chain_var_y) \
        / first_n_samples / pooled_within_chain_var_y

    return (gelman_rubin_statistic_x, gelman_rubin_statistic_y)


def plot_results(results):
    x_values = np.array(results)[:, 0]
    y_values = np.array(results)[:, 1]

    # plot the distribution of x from samples and the expected distribution of x
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title(f"Histogram of Betabinomial (a={a}, b={b}, n={n})")
    x = np.arange(n + 1)
    expected_density = st.betabinom.pmf(x, n, a, b)
    plt.hist(x_values, density=True, bins=x)
    plt.plot(x, expected_density, "purple", label="betabinomial pmf")
    plt.show()

    # plot the distribution of y from samples and the expected distribution of y
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Y")
    plt.ylabel("Density")
    plt.title(f"Histogram of Betabinomial (a={a}, b={b}, n={n})")
    y = np.linspace(0, 0.9, 100)
    expected_density = st.beta.pdf(y, a, b)
    plt.hist(y_values, density=True, bins=30, color="seagreen")
    plt.plot(y, expected_density, "purple", label="betabinomial pmf")
    plt.show()


def plot_gr_method(gelman_rubin_li):
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    x = np.arange(sample_num - min_num_for_averaging + 1)
    plt.plot(x, gelman_rubin_li[:, 0], "b", label="R for X")
    plt.plot(x, gelman_rubin_li[:, 1], "seagreen", label="R for Y")
    plt.xlabel("Iteration")
    plt.ylabel("R")
    plt.hlines(xmin=0, xmax=(sample_num - min_num_for_averaging), y=1.1,
               linestyles="dashdot", colors="black")
    plt.legend()
    plt.title("Convergence Plot by Using the Gelman-Rubin Method")
    plt.show()

run_gibbs_sampling_and_convergence_test()
