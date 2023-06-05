import stan
from pandas import read_csv
import os
import numpy as np
import pickle

os.chdir(os.path.dirname(__file__))
BIRTHWEIGHT_DATA_PATH = "./birthweight.csv"

with open(BIRTHWEIGHT_DATA_PATH) as f:
    data = read_csv(f)

schools_code = """
data {
  int<lower=0> N;
  array[N] real x;
  array[N] real y;
}
parameters {
  real mu1;
  real mu2;
  real<lower=0> tau_2;
  real<lower=0> sigma_2;
  real eta;
}
transformed parameters {
    real<lower=0> tau = sqrt(tau_2);
    real<lower=0> sigma = sqrt(sigma_2);
}

model {
    x ~ normal(mu1, sigma);
    y ~ normal(mu2, sigma);
    mu1 ~ normal(eta, tau);
    mu2 ~ normal(eta, tau);
    // tau_2 ~ inv_gamma(3, 1);
    tau_2 ~ cauchy(0, 0.1);
    sigma_2 ~ cauchy(0, 0.1);
}
"""
m_data = list(data[data["gender"] == 0]["weight"] / 1000)
f_data = list(data[data["gender"] == 1]["weight"] / 1000)

schools_data = {"N": 12,
                "x": m_data,
                "y": f_data}

posterior = stan.build(schools_code, data=schools_data, random_seed=1)
fit_li = posterior.sample(num_chains=5, num_samples=2000)


with open("./result_cauchy_cauchy.pickle", "wb") as f:
    mu1 = np.array_split(fit_li["mu1"][0], 5)
    mu2 = np.array_split(fit_li["mu2"][0], 5)
    tau_2 = np.array_split(fit_li["tau_2"][0], 5)
    simga_2 = np.array_split(fit_li["sigma_2"][0], 5)
    eta = np.array_split(fit_li["eta"][0], 5)
    
    res = {"mu1": mu1,
           "mu2": mu2,
           "tau_2": tau_2,
           "sigma_2": simga_2,
           "eta": eta}
    pickle.dump(res, f)

