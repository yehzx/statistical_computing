import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from pandas import read_csv
from scipy.special import gamma

birthweight_data_path = "./birthweight.csv"


class Distribution():
    def generate_inverse_gamma(a, b):
        return float(1 / np.random.gamma(shape=a, scale=b, size=1))

    def cal_trun_cauchy_density(x, b):
        assert b > 0, "invalid trun_cauchy parameter b"
        return 2 / np.pi / (b * (1 + (x/b)**2)) if x >= 0 else 0

    def cal_inverse_gamma_density(x, a, b):
        return b**a * np.exp(-b / x) * x**(-a-1) / gamma(a)

    def cal_chi_2_density(target, param):
        return st.chi2.pdf(target, df=param)

    def generate_sample_from_chi_2(df):
        return np.random.chisquare(df)

# class MetroHast():



class BirthWeightData():
    def __init__(self, sample_size):
        self.sample_size = sample_size

        # hyperparam
        self.b_tau = 1
        self.b_sigma = 1
        self.c_tau_max = Distribution.cal_trun_cauchy_density(0, self.b_tau)
        self.c_sigma_max = Distribution.cal_trun_cauchy_density(
            0, self.b_sigma)

        self.configure_initial_values()
        self.configure_gibbs_sampler_tracking()

    def configure_initial_values(self):
        with open(birthweight_data_path) as f:
            data = read_csv(f)
        self.y = data["weight"]
        self.mu1 = np.mean(data[data["gender" == 0]])
        self.mu2 = np.mean(data[data["gender" == 1]])
        self.eta = (self.mu1 + self.mu2) / 2
        self.tau_2 = np.var([self.mu1, self.mu2])
        self.sigma_2 = np.var(data[data["gender" == 0]]["weight"]) / 2 \
            + np.var(data[data["gender" == 1]]["weight"]) / 2
        self.initial_tau_2 = self.tau_2
        self.initial_sigma_2 = self.sigma_2
        self.n = 24
        self.ni = 12

    def configure_gibbs_sampler_tracking(self):
        self.y_li = [self.y]
        self.mu1_li = [self.mu1]
        self.mu2_li = [self.mu2]
        self.eta_li = [self.eta]
        self.tau_2_li = [self.tau_2]
        self.sigma_2_li = [self.sigma_2]

    def generate_mu(self):
        mean = (np.mean(self.y) / (self.sigma_2/self.ni) + self.eta / self.tau_2) \
            / ((self.ni/self.sigma_2) + 1 / self.tau_2)
        variance = (self.ni/self.sigma_2) + 1 / self.tau_2
        return np.random.normal(loc=mean, scale=np.sqrt(variance))

    def generate_eta(self):
        mean = (self.mu1 + self.mu2) / 2
        variance = self.tau_2 / 2
        return np.random.normal(loc=mean, scale=np.sqrt(variance))

    def generate_sigma_2(self):

        return

    def generate_tau_2(self):
        count = 0
        self.scale = ((self.mu1-self.eta)**2 + (self.mu2-self.eta)**2) / 2

        c_tau_max = Distribution.cal_trun_cauchy_density(0, self.b_tau)
        # AR sampling
        while count < self.sample_size:
            # conditional tau_2
            proposed_tau_2 = Distribution.generate_inverse_gamma(1, self.scale)
            acceptance_prob = \
                Distribution.cal_trun_cauchy_density(
                    proposed_tau_2, self.b_tau) / c_tau_max
            if acceptance_prob > np.random.uniform():
                count += 1
                yield proposed_tau_2

        # MH sampling
        samples_mh_li, samples_imh_li = self.run_metropolis_hastings_sampler()

        return True

    def run_metropolis_hastings_sampler(self):
        samples_mh_li = self.metropolis_hastings_sampler(2 * self.sample_size)
        samples_imh_li = self.independent_metropolis_hastings_sampler(
            2 * self.sample_size)

        # discard first sample_num/2 samples
        samples_mh_li = samples_mh_li[self.sample_size:]
        samples_imh_li = samples_imh_li[self.sample_size:]

        return (samples_mh_li, samples_imh_li)

    def metropolis_hastings_sampler(self):
        samples_li = []
        samples_li.append(Distribution.generate_sample_from_chi_2(self.tau_2))

        for i in range(1, self.sample_size):
            last_sample = samples_li[i - 1]
            this_sample = Distribution.generate_sample_from_chi_2(last_sample)
            if self.determine_accept_or_not(this_sample, last_sample):
                samples_li.append(this_sample)
            else:
                samples_li.append(last_sample)

        return samples_li

    def determine_accept_or_not(self, this_sample, last_sample):
        threshold = np.random.uniform()
        accepted_prob = np.min([1,
            self.cal_tau_2_density(this_sample)
            * Distribution.cal_chi_2_density(last_sample, this_sample)
            / self.cal_tau_2_density(last_sample)
            / Distribution.cal_chi_2_density(this_sample, last_sample)])

        return True if threshold <= accepted_prob else False

    def cal_tau_2_density(self):
        density = Distribution.cal_inverse_gamma_density(self.tau_2, 1, self.scale) \
            * Distribution.cal_trun_cauchy_density(self.tau_2, self.b_tau)
        
        return density

    def independent_metropolis_hastings_sampler(self):
        samples_li = []
        samples_li.append(Distribution.generate_sample_from_chi_2(self.initial_tau_2))

        for i in range(1, self.sample_size):
            last_sample = samples_li[i - 1]
            this_sample = Distribution.generate_sample_from_chi_2(self.initial_tau_2)
            if self.determine_accept_or_not_ind(this_sample, last_sample):
                samples_li.append(this_sample)
            else:
                samples_li.append(last_sample)

        return samples_li

    def determine_accept_or_not_ind(self, this_sample, last_sample):
        threshold = np.random.uniform()
        # incorrect?
        """
        accepted_prob = np.min([1,
                                cal_trun_cauchy_density(this_sample, b_tau)
                                / cal_trun_cauchy_density(last_sample, b_tau)])
        """
        accepted_prob = np.min([1,
            self.cal_tau_2_density(this_sample)
            * Distribution.cal_chi_2_density(last_sample, 2)
            / self.cal_tau_2_density(last_sample)
            / Distribution.cal_chi_2_density(this_sample, 2)])
        
        return True if threshold <= accepted_prob else False





# x = np.linspace(0, 4, 100)
# density = st.rayleigh.pdf(x, scale=sigma)

samples_li = generate_inverse_gamma(1, 1, 1000)
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 14})
# plt.plot(x, density, "b", label="rayleigh pdf")
plt.xlabel("X")
plt.ylabel("Density")
# plt.title("Histogram of Rayleigh(0.5)")
plt.hist(samples_li, density=True, alpha=0.3, color="b")
plt.show()

