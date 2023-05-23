import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

"""
1. Use the Metropolis-Hastings sampler to generate a sample from a Rayleigh
distribution with the proposal distribution Y ∼ Gamma(Xt, 1)
"""
n = 12000
initial_x = 1
sigma = 0.5

np.random.seed(1000)


def run_metropolis_hastings_and_plot_result():
    samples_li = metropolis_hastings_sampler()

    # discard first 2000 samples
    samples_li = samples_li[2000:]

    plot_result(samples_li)


def metropolis_hastings_sampler():
    samples_li = []
    samples_li.append(generate_initial_sample_from_gamma())

    for i in range(1, n):
        last_sample = samples_li[i - 1]
        this_sample = generate_sample_from_gamma(last_sample)
        if determine_accept_or_not(this_sample, last_sample):
            samples_li.append(this_sample)
        else:
            samples_li.append(last_sample)

    return samples_li


def generate_initial_sample_from_gamma():
    return np.random.gamma(initial_x, 1)


def generate_sample_from_gamma(x):
    return np.random.gamma(x, 1)


def determine_accept_or_not(this_sample, last_sample):
    threshold = np.random.uniform()
    accpeted_prob = np.min([1,
                            cal_rayleigh_density(this_sample)
                            * cal_gamma_density(last_sample, this_sample)
                            / cal_rayleigh_density(last_sample)
                            / cal_gamma_density(this_sample, last_sample)])

    return True if threshold <= accpeted_prob else False


def cal_rayleigh_density(x):
    return x / (sigma**2) * np.exp(-x**2 / (2 * (sigma**2)))


def cal_gamma_density(target, param):
    return st.gamma.pdf(target, param, scale=1)


def plot_result(samples_li):
    x = np.linspace(0, 4, 100)
    density = st.rayleigh.pdf(x, scale=sigma)

    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(x, density, "b", label="rayleigh pdf")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.title("Histogram of Rayleigh(0.5)")
    plt.hist(samples_li, density=True, bins=30)
    plt.show()


run_metropolis_hastings_and_plot_result()