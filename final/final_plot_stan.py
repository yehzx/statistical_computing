import matplotlib.pyplot as plt
import os
import pickle
import scienceplots
import matplotlib as mpl
import seaborn as sns
from typing import Literal
from final_main import plot_gr_method, gelman_rubin_method
import numpy as np
plt.style.use("science")


with open("./result_cauchy_cauchy.pickle", "rb") as f:
    res_li = pickle.load(f)

tau_2_li = [res_li["tau_2"][i] for i in range(5)]
sigma_2_li = [res_li["sigma_2"][i] for i in range(5)]
mu1_li = [res_li["mu1"][i] for i in range(5)]
mu2_li = [res_li["mu2"][i] for i in range(5)]
eta_li = [res_li["eta"][i] for i in range(5)]

def plot_stan_result(param_li, param_name, type: Literal["trace", "distribution"]):
    plt.clf()
    plt.figure(figsize=(8, 8))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumseagreen", "goldenrod", "mediumpurple", "cornflowerblue", "indianred"])
    plt.rcParams.update({'font.size': 20})

    if type == "trace":
        plt.ylabel(param_name)
        plt.xlabel("Iteration")
        for idx, param_res in enumerate(param_li):
            plt.plot(param_res, label=f"chain {idx}")
    elif type == "distribution":
        plt.ylabel("Density")
        plt.xlabel(param_name)
        for idx, param_res in enumerate(param_li):
            sns.kdeplot(param_res, bw_adjust=0.3, label=f"chain {idx}")
    # plt.legend()
    plt.show()

def run_plot_gr_method(result_li, label):
    gelman_rubin_statistic_li = np.array([gelman_rubin_method(i, result_li) for i in range(10, 2000 + 1)])
    plot_gr_method(gelman_rubin_statistic_li, label)

plot_stan_result(tau_2_li, "tau_2", "trace")
plot_stan_result(sigma_2_li, "sigma_2", "trace")
plot_stan_result(mu1_li, "mu1", "trace")
plot_stan_result(mu2_li, "mu2", "trace")
plot_stan_result(eta_li, "eta", "trace")

plot_stan_result(tau_2_li, "tau_2", "distribution")
plot_stan_result(sigma_2_li, "sigma_2", "distribution")
plot_stan_result(mu1_li, "mu1", "distribution")
plot_stan_result(mu2_li, "mu2", "distribution")
plot_stan_result(eta_li, "eta", "distribution")

run_plot_gr_method(tau_2_li, "tau_2")
run_plot_gr_method(sigma_2_li, "sigma_2")
run_plot_gr_method(mu1_li, "mu1")
run_plot_gr_method(mu2_li, "mu2")
run_plot_gr_method(eta_li, "eta")
# plt.plot(range(2000), tau_2)
# plt.hist(tau_2, bins=30)
# plt.plot(range(2000), eta)
# plt.hist(eta,density=True, bins=30)
# plt.hist(mu1,density=True, bins=30)
# plt.hist(sigma_2,density=True, bins=30)