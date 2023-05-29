from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from pandas import read_csv
from scipy.special import gamma

import os
os.chdir(os.path.dirname(__file__))
BIRTHWEIGHT_DATA_PATH = "./birthweight.csv"


class Distribution():
    @staticmethod
    def generate_inverse_gamma(a, b):
        return float(1 / np.random.gamma(shape=a, scale=b, size=1))

    @staticmethod
    def cal_trun_cauchy_density(x, b):
        assert b > 0, "invalid trun_cauchy parameter b"
        return 2 / np.pi / (b * (1 + (x/b)**2)) if x >= 0 else 0

    @staticmethod
    def cal_inverse_gamma_density(x, a, b):
        return b**a * np.exp(-b / x) * x**(-a-1) / gamma(a)

    @staticmethod
    def cal_chi_2_density(target, param):
        return st.chi2.pdf(target, df=param)

    @staticmethod
    def generate_chi_2(df):
        return np.random.chisquare(df)


class Sampler():
    def run_metropolis_hastings_sampler(self, sample_size,
                                        method: Literal["mcmc", "independent"],
                                        target: Literal["tau_2", "sigma_2"],
                                        burnin=None):
        self.target = target
        self.method = method
        self.configure_metropolis_hastings(sample_size, burnin)

        if method == "mcmc":
            samples_li = self.metropolis_hastings_sampler(target)
        elif method == "independent":
            samples_li = self.independent_metropolis_hastings_sampler(target)
        else:
            raise Exception("Invalid method")
        
        samples_li = samples_li[-self.sample_size:]

        return samples_li

    def configure_metropolis_hastings(self, sample_size, burnin):
        self.running_size = 2 * sample_size if sample_size >= 25 else 50
        self.burnin = sample_size if burnin == None else burnin
        self.sample_size = sample_size

    def metropolis_hastings_sampler(self):
        samples_li = []
        sampling_distribution = Distribution.generate_chi_2
        if self.target == "tau_2":
            samples_li.append(sampling_distribution(BirthWeightData.tau_2))
        elif self.target == "sigma_2":
            samples_li.append(sampling_distribution(BirthWeightData.sigma_2))

        for i in range(1, self.running_size):
            last_sample = samples_li[i - 1]
            this_sample = sampling_distribution(last_sample)
            if self.determine_accept_or_not(this_sample, last_sample):
                samples_li.append(this_sample)
            else:
                samples_li.append(last_sample)

        return samples_li

    def determine_accept_or_not(self, this_sample, last_sample):
        threshold = np.random.uniform()
        if self.target == "tau_2":
            accepted_prob = np.min([1,
                BirthWeightData.cal_tau_2_density(this_sample)
                * Distribution.cal_chi_2_density(last_sample, this_sample)
                / BirthWeightData.cal_tau_2_density(last_sample)
                / Distribution.cal_chi_2_density(this_sample, last_sample)])
        elif self.target == "sigma_2":
            accepted_prob = np.min([1,
                BirthWeightData.cal_sigma_2_density(this_sample)
                * Distribution.cal_chi_2_density(last_sample, this_sample)
                / BirthWeightData.cal_sigma_2_density(last_sample)
                / Distribution.cal_chi_2_density(this_sample, last_sample)])

        return True if threshold <= accepted_prob else False

    def independent_metropolis_hastings_sampler(self):
        samples_li = []
        sampling_distribution = Distribution.generate_inverse_gamma
        if self.target == "tau_2":
            samples_li.append(sampling_distribution(BirthWeightData.initial_tau_2))
        elif self.target == "sigma_2":
            samples_li.append(sampling_distribution(BirthWeightData.initial_sigma_2))
        for i in range(1, self.running_size):
            last_sample = samples_li[i - 1]
            this_sample = sampling_distribution(
                BirthWeightData.initial_tau_2)
            if self.determine_accept_or_not_ind(this_sample, last_sample):
                samples_li.append(this_sample)
            else:
                samples_li.append(last_sample)

        return samples_li

    def determine_accept_or_not_ind(self, this_sample, last_sample):
        threshold = np.random.uniform()
        target_density = Distribution.cal_trun_cauchy_density

        if self.target == "tau_2":
            accepted_prob = np.min([1,
                target_density(this_sample, BirthWeightData.b_tau)
                / target_density(last_sample, BirthWeightData.b_tau)])
        elif self.target == "sigma_2":
            accepted_prob = np.min([1,
                target_density(this_sample, BirthWeightData.b_sigma)
                / target_density(last_sample, BirthWeightData.b_sigma)])

        return True if threshold <= accepted_prob else False

    @staticmethod
    def acceptance_rejection_sampler(sample_size,
                                     target: Literal["tau_2", "sigma_2"]):
        count = 0
        samples_li = []
        while count < sample_size:
            if target == "tau_2":
                proposed = Distribution.generate_inverse_gamma(
                    1, BirthWeightData.scale_tau)
                acceptance_prob = \
                    Distribution.cal_trun_cauchy_density(
                        proposed, BirthWeightData.b_tau) \
                    / BirthWeightData.c_tau_max
            elif target == "sigma_2":
                proposed = Distribution.generate_inverse_gamma(
                    BirthWeightData.n / 2 - 1, BirthWeightData.scale_sigma)
                acceptance_prob = \
                    Distribution.cal_trun_cauchy_density(
                        proposed, BirthWeightData.b_sigma) \
                    / BirthWeightData.c_sigma_max

            if acceptance_prob > np.random.uniform():
                count += 1
                samples_li.append(proposed)

        return samples_li

class BirthWeightData():
    def __init__(self, sample_size, method: Literal["acceptance-rejection",
                                                    "independent metropolis-hastings",
                                                    "metroplis-hastings"]):
        self.sample_size = sample_size
        self.method = method

        # hyperparam
        # BirthWeightData.a_tau = 2
        BirthWeightData.b_tau = 1
        BirthWeightData.b_sigma = 1
        BirthWeightData.c_tau_max = Distribution.cal_trun_cauchy_density(0, self.b_tau)
        BirthWeightData.c_sigma_max = Distribution.cal_trun_cauchy_density(0, self.b_sigma)

        self.configure_initial_values()
        self.configure_gibbs_sampler_tracking()
        self.start_gibbs_sampling()

    def configure_initial_values(self):
        with open(BIRTHWEIGHT_DATA_PATH) as f:
            data = read_csv(f)
        BirthWeightData.y = data["weight"]
        BirthWeightData.mu1 = np.mean(data[data["gender"] == 1]["weight"])
        BirthWeightData.mu2 = np.mean(data[data["gender"] == 1]["weight"])
        BirthWeightData.eta = (self.mu1 + self.mu2) / 2
        BirthWeightData.tau_2 = np.var([self.mu1, self.mu2])
        BirthWeightData.sigma_2 = np.var(data[data["gender"] == 0]["weight"]) / 2 \
            + np.var(data[data["gender"] == 1]["weight"]) / 2
        BirthWeightData.initial_tau_2 = self.tau_2
        BirthWeightData.initial_sigma_2 = self.sigma_2
        BirthWeightData.n = 24
        BirthWeightData.ni = 12

    def configure_gibbs_sampler_tracking(self):
        self.y_li = [self.y]
        self.mu1_li = [self.mu1]
        self.mu2_li = [self.mu2]
        self.eta_li = [self.eta]
        self.sigma_2_li = [self.sigma_2]
        self.tau_2_li = [self.tau_2]
    
    def start_gibbs_sampling(self):
        for i in range(self.sample_size):
            self.mu1 = self.generate_mu("mu1")
            self.mu2 = self.generate_mu("mu2")
            self.eta = self.generate_eta()
            self.sigma_2 = self.generate_sigma_2()
            self.tau_2 = self.generate_tau_2()
            self.track_param()
            self.y = self.generate_y()
            self.track_data()

    def generate_mu(self, target):
        if target == "mu1":
            y_mean = np.mean(self.y[:self.ni])
        elif target == "mu2":
            y_mean = np.mean(self.y[self.ni:])

        mean = (y_mean / (self.sigma_2/self.ni) + self.eta / self.tau_2) \
            / ((self.ni/self.sigma_2) + 1 / self.tau_2)
        variance = 1 / ((self.ni/self.sigma_2) + 1 / self.tau_2)

        return np.random.normal(loc=mean, scale=np.sqrt(variance))

    def generate_eta(self):
        mean = (self.mu1 + self.mu2) / 2
        variance = self.tau_2 / 2

        return np.random.normal(loc=mean, scale=np.sqrt(variance))

    def generate_sigma_2(self):
        BirthWeightData.scale_sigma = np.sum((self.y[:self.ni]-self.mu1)**2 / 2) \
                           + np.sum((self.y[self.ni:]-self.mu2)**2 / 2)
        
        
        if self.method == "acceptance-rejection":
            sample_li = Sampler.acceptance_rejection_sampler(
                sample_size=1, target="sigma_2")
        elif self.method == "independent metropolis-hastings":
            sample_li = Sampler.run_metropolis_hastings_sampler(
                sample_size=1, method="independent", target="sigma_2")
        elif self.method == "metroplis-hastings":
            sample_li = Sampler.run_metropolis_hastings_sampler(
                sample_size=1, method="mcmc", target="sigma_2")
        else:
            raise Exception("Invalid sampler name")
        
        return float(sample_li)

    def generate_tau_2(self):
        BirthWeightData.scale_tau = ((self.mu1-self.eta)**2 + (self.mu2-self.eta)**2) / 2

        if self.method == "acceptance-rejection":
            sample_li = Sampler.acceptance_rejection_sampler(
                sample_size=1, target="tau_2")
        elif self.method == "independent metropolis-hastings":
            sample_li = Sampler.run_metropolis_hastings_sampler(
                sample_size=1, method="independent", target="tau_2")
        elif self.method == "metroplis-hastings":
            sample_li = Sampler.run_metropolis_hastings_sampler(
                sample_size=1, method="mcmc", target="tau_2")
        else:
            raise Exception("Invalid sampler name")

        return float(sample_li)

    def track_param(self):
        self.mu1_li.append(self.mu1)
        self.mu2_li.append(self.mu2)
        self.eta_li.append(self.eta)
        self.sigma_2_li.append(self.sigma_2)
        self.tau_2_li.append(self.tau_2)

    def generate_y(self):
        group1_y = np.random.normal(self.mu1, self.sigma_2, size=self.ni)
        group2_y = np.random.normal(self.mu2, self.sigma_2, size=self.ni)
        
        return np.concatenate((group1_y, group2_y))

    def track_data(self):
        self.y_li.append(self.y)
    
    def cal_tau_2_density(self):
        density = Distribution.cal_inverse_gamma_density(self.tau_2, 1, self.scale_tau) \
            * Distribution.cal_trun_cauchy_density(self.tau_2, self.b_tau)
        
        return density
    
    def cal_sigma_2_density(self):
        density = Distribution.cal_inverse_gamma_density(self.sigma_2, self.n / 2 - 1, self.scale_sigma) \
            * Distribution.cal_trun_cauchy_density(self.sigma_2, self.b_sigma)
        
        return density



ar_result = BirthWeightData(sample_size=50, method="acceptance-rejection")
imh_result = BirthWeightData(sample_size=50, method="independent metropolis-hastings")
mh_result = BirthWeightData(sample_size=50, method="metropolis-hastings")


# x = np.linspace(0, 4, 100)
# density = st.rayleigh.pdf(x, scale=sigma)

# samples_li = generate_inverse_gamma(1, 1, 1000)
# plt.figure(figsize=(8, 8))
# plt.rcParams.update({'font.size': 14})
# # plt.plot(x, density, "b", label="rayleigh pdf")
# plt.xlabel("X")
# plt.ylabel("Density")
# # plt.title("Histogram of Rayleigh(0.5)")
# plt.hist(samples_li, density=True, alpha=0.3, color="b")
# plt.show()

