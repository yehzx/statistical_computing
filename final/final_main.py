import os
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy.stats as st
import seaborn as sns
from pandas import read_csv
from scipy.special import gamma

import final_distribution as distribution
from tqdm import tqdm

plt.style.use("science")
os.chdir(os.path.dirname(__file__))
BIRTHWEIGHT_DATA_PATH = "./birthweight.csv"

SAMPLE = 2000
min_num_for_averaging = 10
# generate 5 chains
CHAIN = 5

class BirthWeightData():
    def __init__(self, sample_size, method: Literal["acceptance-rejection",
                                                    "independent metropolis-hastings",
                                                    "metroplis-hastings"],
                                    joint_pdf: Literal["cauchy_cauchy", "invgamma_cauchy"]):
        self.sample_size = sample_size
        self.method = method
        self.distribution = joint_pdf

        # hyperparam
        BirthWeightData.a_tau_inv = 3
        BirthWeightData.b_tau_inv = 1
        if joint_pdf == "invgamma_cauchy":
            BirthWeightData.b_sigma_cau = 0.1
        elif joint_pdf == "cauchy_cauchy":
            BirthWeightData.b_tau_cau = 0.1
            BirthWeightData.b_sigma_cau = 0.1
        BirthWeightData.c_sigma_max = distribution.cal_trun_cauchy_density(0, self.b_sigma_cau)

        self.configure_initial_values()
        self.configure_gibbs_sampler_tracking()
        self.start_gibbs_sampling()

    def configure_initial_values(self):
        with open(BIRTHWEIGHT_DATA_PATH) as f:
            data = read_csv(f)
        
        # scaling
        if self.method in ("acceptance-rejection", "independent metropolis-hastings"):
            self.scaling = 1000
        else:
            self.scaling = 1000
        
        m_data = data[data["gender"] == 0] / self.scaling
        f_data = data[data["gender"] == 1] / self.scaling
        
        BirthWeightData.y = np.concatenate((m_data["weight"], f_data["weight"]))
        BirthWeightData.mu1 = np.mean(m_data["weight"])
        BirthWeightData.mu2 = np.mean(f_data["weight"])
        BirthWeightData.eta = (self.mu1 + self.mu2) / 2
        BirthWeightData.tau_2 = np.var([self.mu1, self.mu2])
        BirthWeightData.sigma_2 = np.var(m_data["weight"]) / 2 \
                                  + np.var(f_data["weight"]) / 2
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
        for i in tqdm(range(self.sample_size)):
            self.mu1 = self.generate_mu("mu1")
            self.mu2 = self.generate_mu("mu2")
            self.eta = self.generate_eta()
            # required distribution
            self.sigma_2 = self.generate_sigma_2()
            if self.distribution == "invgamma_cauchy":
                self.tau_2 = self.generate_tau_2_v2()
            elif self.distribution == "cauchy_cauchy":
                self.tau_2 = self.generate_tau_2()

            self.track_param()
            # self.y = self.generate_y()
            # self.track_data()

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
            sample_li = acceptance_rejection_sampler(sample_size=1, target="sigma_2")
        elif self.method == "independent metropolis-hastings":
            sample_li = MetroHasting(sample_size=1, method="independent", target="sigma_2", joint_pdf=self.distribution).samples_li
        elif self.method == "metropolis-hastings":
            sample_li = MetroHasting(sample_size=1, method="mcmc", target="sigma_2", joint_pdf=self.distribution).samples_li
        else:
            raise Exception("Invalid sampler name")
        
        return float(sample_li[0])

    def generate_tau_2(self):
        if self.distribution == "invgamma_cauchy":
            BirthWeightData.scale_tau = ((self.mu1-self.eta)**2 + (self.mu2-self.eta)**2) / 2 + self.b_tau_inv
        elif self.distribution == "cauchy_cauchy":
            BirthWeightData.scale_tau = ((self.mu1-self.eta)**2 + (self.mu2-self.eta)**2) / 2

        if self.method == "acceptance-rejection":
            sample_li = acceptance_rejection_sampler(sample_size=1, target="tau_2")
        elif self.method == "independent metropolis-hastings":
            sample_li = MetroHasting(sample_size=1, method="independent", target="tau_2", joint_pdf=self.distribution).samples_li
        elif self.method == "metropolis-hastings":
            sample_li = MetroHasting(sample_size=1, method="mcmc", target="tau_2", joint_pdf=self.distribution).samples_li
        else:
            raise Exception("Invalid sampler name")

        return float(sample_li[0])
    
    def generate_tau_2_v2(self):
        scale_tau = ((self.mu1-self.eta)**2 + (self.mu2-self.eta)**2) / 2 + self.b_tau_inv
        
        return distribution.generate_inverse_gamma(self.a_tau_inv + 1, scale_tau)

    def track_param(self):
        self.mu1_li.append(self.mu1)
        self.mu2_li.append(self.mu2)
        self.eta_li.append(self.eta)
        self.sigma_2_li.append(self.sigma_2)
        self.tau_2_li.append(self.tau_2)

    def generate_y(self):
        group1_y = np.random.normal(self.mu1, np.sqrt(self.sigma_2), size=self.ni)
        group2_y = np.random.normal(self.mu2, np.sqrt(self.sigma_2), size=self.ni)
        
        return np.concatenate((group1_y, group2_y))

    def track_data(self):
        self.y_li.append(self.y)
    

class MetroHasting():
    def __init__(self, sample_size, method: Literal["mcmc", "independent"],
                 target: Literal["tau_2", "sigma_2"], joint_pdf, burnin=None):
        self.target = target
        self.method = method
        self.distribution = joint_pdf
        self.configure_metropolis_hastings(sample_size, burnin)
        self.run_metropolis_hastings()

    def run_metropolis_hastings(self):
        if self.method == "mcmc":
            samples_li = self.metropolis_hastings_sampler()
        elif self.method == "independent":
            samples_li = self.independent_metropolis_hastings_sampler()
        else:
            raise Exception("Invalid method")
        
        self.samples_li = samples_li[-self.sample_size:]

    def configure_metropolis_hastings(self, sample_size, burnin):
        self.running_size = 2 * sample_size if sample_size >= 25 else 50
        self.burnin = sample_size if burnin == None else burnin
        self.sample_size = sample_size

    def metropolis_hastings_sampler(self):
        samples_li = []
        sampling_distribution = distribution.generate_chi_2
        if self.target == "tau_2":
            samples_li.append(BirthWeightData.tau_2)
            # samples_li.append(sampling_distribution(BirthWeightData.tau_2))
        elif self.target == "sigma_2":
            samples_li.append(BirthWeightData.sigma_2)
            # samples_li.append(sampling_distribution(BirthWeightData.sigma_2))

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
            try:
                accepted_prob = cal_tau_2_density(this_sample, self.distribution) \
                * distribution.cal_chi_2_density(last_sample, this_sample) \
                / cal_tau_2_density(last_sample, self.distribution) \
                / distribution.cal_chi_2_density(this_sample, last_sample)
            except FloatingPointError:
                accepted_prob = 0
            accepted_prob = np.min([1, accepted_prob])
        elif self.target == "sigma_2":
            try:
                accepted_prob = cal_sigma_2_density(this_sample) \
                * distribution.cal_chi_2_density(last_sample, this_sample) \
                / cal_sigma_2_density(last_sample) \
                / distribution.cal_chi_2_density(this_sample, last_sample)
            except FloatingPointError:
                accepted_prob = 0
            except ZeroDivisionError:
                accepted_prob = 1

            accepted_prob = np.min([1, accepted_prob])

        return True if threshold <= accepted_prob else False

    def independent_metropolis_hastings_sampler(self):
        samples_li = []
        sampling_distribution = distribution.generate_inverse_gamma
        if self.target == "tau_2":
            samples_li.append(sampling_distribution(BirthWeightData.a_tau_inv + 1, BirthWeightData.scale_tau))
            # samples_li.append(BirthWeightData.tau_2)
        elif self.target == "sigma_2":
            samples_li.append(sampling_distribution(BirthWeightData.n / 2 - 1, BirthWeightData.scale_sigma))
            # samples_li.append(BirthWeightData.sigma_2)

        for i in range(1, self.running_size):
            last_sample = samples_li[i - 1]
            if self.target == "sigma_2":
                this_sample = sampling_distribution(BirthWeightData.n / 2 - 1, BirthWeightData.scale_sigma)
            elif self.target == "tau_2":
                pass
                # this_sample = sampling_distribution(BirthWeightData.a_tau_inv + 1, BirthWeightData.scale_tau)
            if self.determine_accept_or_not_ind(this_sample, last_sample):
                samples_li.append(this_sample)
            else:
                samples_li.append(last_sample)

        return samples_li

    def determine_accept_or_not_ind(self, this_sample, last_sample):
        threshold = np.random.uniform()
        target_density = distribution.cal_trun_cauchy_density

        if self.target == "tau_2":
            accepted_prob = np.min([1,
                target_density(this_sample, BirthWeightData.b_tau_cau)
                / target_density(last_sample, BirthWeightData.b_tau_cau)])
        elif self.target == "sigma_2":
            accepted_prob = np.min([1,
                target_density(this_sample, BirthWeightData.b_sigma_cau)
                / target_density(last_sample, BirthWeightData.b_sigma_cau)])

        return True if threshold <= accepted_prob else False

def acceptance_rejection_sampler(sample_size,
                                 target: Literal["tau_2", "sigma_2"]):
    count = 0
    samples_li = []
    while count < sample_size:
        if target == "sigma_2":
            proposed = distribution.generate_inverse_gamma(
                BirthWeightData.n / 2 - 1, BirthWeightData.scale_sigma)
            acceptance_prob = \
                distribution.cal_trun_cauchy_density(
                    proposed, BirthWeightData.b_sigma_cau) \
                / BirthWeightData.c_sigma_max
        if acceptance_prob > np.random.uniform():
            count += 1
            samples_li.append(proposed)

    return samples_li

def cal_tau_2_density(tau_2, dist):
    if dist == "cauchy_cauchy":
        try:
            density = np.exp(-BirthWeightData.scale_tau / tau_2) / tau_2 \
                * distribution.cal_trun_cauchy_density(tau_2, BirthWeightData.b_tau_cau)
        except (ZeroDivisionError, FloatingPointError):
            return 0


    else:
        density = distribution.cal_inverse_gamma_density(tau_2, BirthWeightData.a_tau_inv + 1, BirthWeightData.scale_tau + BirthWeightData.b_tau_inv)
    return density

def cal_sigma_2_density(sigma_2):
    density = distribution.cal_inverse_gamma_density(sigma_2, BirthWeightData.n / 2 - 1, BirthWeightData.scale_sigma) \
        * distribution.cal_trun_cauchy_density(sigma_2, BirthWeightData.b_sigma_cau)
    
    return density


def plot_result_y(result: BirthWeightData):
    plt.clf()
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 14})
    scaling_factor = 100
    male = ["Male"] * 12
    female = ["Female"] * 12
    for i, color in zip(range(1, 4), ("seagreen", "b", "violet")):
        plt.scatter(male, result.y_li[-i][:12] * scaling_factor, alpha=0.3, color=color)
        plt.scatter(female, result.y_li[-i][12:] * scaling_factor, alpha=0.3, color=color)
    plt.xlabel("Birthweight (g)")
    plt.ylabel("Gender")
    plt.show()


def plot_result_param(result, type: Literal["trace", "distribution"], single=True):
    if not single:
        result[0].mu1_li = [result[i].mu1_li for i in range(CHAIN)]
        result[0].mu2_li = [result[i].mu2_li for i in range(CHAIN)]
        result[0].eta_li = [result[i].eta_li for i in range(CHAIN)]
        result[0].sigma_2_li = [result[i].sigma_2_li for i in range(CHAIN)]
        result[0].tau_2_li = [result[i].tau_2_li for i in range(CHAIN)]
        result = result[0]        

    for label, param_result in zip(("mu1", "mu2", "eta", "sigma_2", "tau_2"), (result.mu1_li, result.mu2_li, result.eta_li, result.sigma_2_li, result.tau_2_li)):
        plt.clf()
        plt.figure(figsize=(8, 8))
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumseagreen", "goldenrod", "mediumpurple", "cornflowerblue", "indianred"])
        plt.rcParams.update({'font.size': 18})

        # if label in ("mu1", "mu2", "eta"):
        #     param_result = np.array(param_result) * 1000
        # plt.hist(param_result, density=True, bins=30, alpha=0.3)

        if single:
            sns.kdeplot(param_result, bw_adjust=0.3, label=f"chain {idx}", color="b")
        else:
            if type == "distribution":
                plt.xlabel(label)
                plt.ylabel("Density")
                for idx, param_res in enumerate(param_result):
                    sns.kdeplot(param_res, bw_adjust=0.3, label=f"chain {idx}")
            elif type == "trace":
                plt.ylabel(label)
                plt.xlabel("Iteration")
                for idx, param_res in enumerate(param_result):
                    plt.plot(param_res, label=f"chain {idx}")
        plt.show()


# def plot_trace_param(result):
#     for label, param_result in zip(("mu1", "mu2", "eta", "sigma_2", "tau_2"), (result.mu1_li, result.mu2_li, result.eta_li, result.sigma_2_li, result.tau_2_li)):
#         plt.clf()
#         plt.figure(figsize=(8, 8))
#         plt.rcParams.update({'font.size': 18})
#         mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumseagreen", "goldenrod", "mediumpurple", "cornflowerblue", "indianred"])

#         plt.ylabel(label)
#         plt.xlabel("Iteration")
#         for idx, param_res in enumerate(param_li):
#             plt.plot(param_res, label=f"chain {idx}")
#         # if label in ("mu1", "mu2", "eta"):
#         #     param_result = np.array(param_result) * 100
#         plt.plot(np.arange(0, SAMPLE+1), param_result)
#         plt.show()

def gelman_rubin_method(first_n_samples, many_results):
    # calculate statistics from the given first n samples
    many_results = np.array([result[:first_n_samples]
                            for result in many_results])
    chain_mean = np.mean(many_results, axis=1)
    between_chain_var = np.var(chain_mean) * first_n_samples

    # many_results[#chain, #sample]
    pooled_within_chain_var = sum(
        [np.var(many_results[i]) for i in range(CHAIN)]) / CHAIN

    gelman_rubin_statistic = \
        ((first_n_samples - 1) * pooled_within_chain_var + between_chain_var) \
        / first_n_samples / pooled_within_chain_var

    return gelman_rubin_statistic

def plot_gr_method(gelman_rubin_li, label):
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 18})
    x = np.arange(SAMPLE - min_num_for_averaging + 1)
    plt.plot(x, gelman_rubin_li, "b", label=f"{label}")
    # plt.plot(x, gelman_rubin_li, "b", label=f"{label}")
    plt.xlabel("Iteration")
    plt.ylabel("R")
    plt.hlines(xmin=0, xmax=(SAMPLE - min_num_for_averaging), y=1.1,
               linestyles="dashdot", colors="black")
    plt.legend()
    plt.title("Convergence Plot by Using the Gelman-Rubin Method")
    plt.show()



def run_plot_gr_method(result_li):
    label_li = ["mu1", "mu2", "eta", "sigma_2", "tau_2"]
    result_mu1_li = [result_li[i].mu1_li for i in range(CHAIN)]
    result_mu2_li = [result_li[i].mu2_li for i in range(CHAIN)]
    result_eta_li = [result_li[i].eta_li for i in range(CHAIN)]
    result_sigma_li = [result_li[i].sigma_2_li for i in range(CHAIN)]
    result_tau_li = [result_li[i].tau_2_li for i in range(CHAIN)]
    result_param_li = [result_mu1_li, result_mu2_li, result_eta_li, result_sigma_li, result_tau_li]

    for label, result_param in zip(label_li, result_param_li): 
        gelman_rubin_statistic_li = np.array([gelman_rubin_method(i, result_param) for i in range(min_num_for_averaging, SAMPLE + 1)])
        plot_gr_method(gelman_rubin_statistic_li, label)
