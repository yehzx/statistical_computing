import numpy as np
from scipy.special import gamma
import scipy.stats as st
import matplotlib.pyplot as plt

b_tau = 1
b_sigma = 1
observation = 100


def generate_tau_2(sample_size):
    count = 0

    c_tau_max = cal_trun_cauchy_density(0, b_tau)
    # AR sampling
    samples_ar_li = []
    while count < sample_size:
        proposed_tau_2 = generate_inverse_gamma(a, b)
        acceptance_prob = \
            cal_trun_cauchy_density(proposed_tau_2, b_tau) / c_tau_max
        if acceptance_prob > np.random.uniform():
            count += 1
            samples_ar_li.append(proposed_tau_2)
    
    # MH sampling
    samples_mh_li, samples_imh_li = run_metropolis_hastings_sampler(sample_size)

    return [samples_ar_li, samples_mh_li, samples_imh_li]

def generate_inverse_gamma(a, b, n=1):
    return float(1 / np.random.gamma(shape=a, scale=b, size=1))

def cal_trun_cauchy_density(x, b):
    assert b > 0, "invalid trun_cauchy parameter b"
    return 2 / np.pi / (b * (1 + (x/b)**2)) if x >= 0 else 0

def run_metropolis_hastings_sampler(sample_size):
    samples1_li = metropolis_hastings_sampler(2 * sample_size)
    samples2_li = independent_metropolis_hastings_sampler(2 * sample_size)

    # discard first sample_num/2 samples
    samples1_li = samples1_li[sample_size:]
    samples2_li = samples2_li[sample_size:]

    return [samples1_li, samples2_li]


def metropolis_hastings_sampler(sample_size):
    samples_li = []
    samples_li.append(generate_sample_from_chi_2(2))

    for i in range(1, sample_size):
        last_sample = samples_li[i - 1]
        this_sample = generate_sample_from_chi_2(last_sample)
        if determine_accept_or_not(this_sample, last_sample):
            samples_li.append(this_sample)
        else:
            samples_li.append(last_sample)

    return samples_li

def generate_sample_from_chi_2(df):
    return np.random.chisquare(df)

def determine_accept_or_not(this_sample, last_sample):
    threshold = np.random.uniform()
    accepted_prob = np.min([1,
                            cal_tau_2_density(this_sample)
                            * cal_chi_2_density(last_sample, this_sample)
                            / cal_tau_2_density(last_sample)
                            / cal_chi_2_density(this_sample, last_sample)])

    return True if threshold <= accepted_prob else False

def cal_tau_2_density(tau_2):
    density = cal_inverse_gamma_density(tau_2, a, b) \
             * cal_trun_cauchy_density(tau_2, b_tau)
    return density


def cal_inverse_gamma_density(x, a, b):
    return b**a * np.exp(-b / x) * x**(-a-1) / gamma(a)

        
def cal_chi_2_density(target, param):
    return st.chi2.pdf(target, df=param)


def independent_metropolis_hastings_sampler(sample_size):
    samples_li = []
    samples_li.append(generate_sample_from_chi_2(2))

    for i in range(1, sample_size):
        last_sample = samples_li[i - 1]
        this_sample = generate_sample_from_chi_2(2)
        if determine_accept_or_not_ind(this_sample, last_sample):
            samples_li.append(this_sample)
        else:
            samples_li.append(last_sample)
    
    return samples_li

def determine_accept_or_not_ind(this_sample, last_sample):
    threshold = np.random.uniform()
    # incorrect?
    """
    accepted_prob = np.min([1,
                            cal_trun_cauchy_density(this_sample, b_tau)
                            / cal_trun_cauchy_density(last_sample, b_tau)])
    """
    accepted_prob = np.min([1,
                            cal_tau_2_density(this_sample)
                            * cal_chi_2_density(last_sample, 2)
                            / cal_tau_2_density(last_sample)
                            / cal_chi_2_density(this_sample, 2)])
    return True if threshold <= accepted_prob else False


a = 3
b = 1
samples_ar_li, samples_mh_li, samples_imh_li = generate_tau_2(1000)

plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 14})
plt.xlabel("X")
plt.ylabel("Density")
bin = np.arange(0, 6, 0.2)
plt.hist(samples_ar_li, density=True, bins=bin, alpha=0.3, color="b", label="ar")
plt.hist(samples_mh_li, density=True, bins=bin, alpha=0.3, color="g", label="mh")
plt.hist(samples_imh_li, density=True, bins=bin, alpha=0.3, color="r", label="imh")
plt.legend()
plt.show()
