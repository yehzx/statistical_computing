import numpy as np
import pandas as pd
import scipy.stats as st

# set seed = 1000
np.random.seed(1000)

data = pd.read_csv("./data.csv")
del data["id"]

def cal_data_theta(data):
    cov_matrix = data.cov(ddof=1)
    Lambda, U = np.linalg.eig(cov_matrix)
    theta_est = Lambda[0] / sum(Lambda)

    return theta_est

result_ori = cal_data_theta(data)

# bootstrap
result_bs = [cal_data_theta(data.sample(n=88, replace=True)) for i in range(1000)]
bias_bs = np.mean(result_bs) - result_ori
stdev_bs = np.std(result_bs, ddof=1)
print(bias_bs, stdev_bs)


# jackknife
result_jk = [cal_data_theta(data.drop([i])) for i in range(88)]
bias_jk = 87 * (np.mean(result_jk) - result_ori)
stdev_jk = np.sqrt(87) * np.std(result_jk)
print(bias_jk, stdev_jk)


# BCa
indicator = sum([result_bs[i] - result_ori < 0 for i in range(88)])
z_0_est = st.norm.ppf(1/88 * indicator)
acc_factor = -st.skew(result_bs) / 6
# equivalent skewness calculation
# m2 = 1/1000 * np.sum((np.mean(result_bs)-result_bs) ** 2)
# m3 = 1/1000 * np.sum((np.mean(result_bs)-result_bs) ** 3)
# acc_factor = m3 / (m2**(3/2)) / 6

adjust_1 = z_0_est + (z_0_est+st.norm.ppf(0.025)) / (1 - acc_factor*(z_0_est+st.norm.ppf(0.025)))
adjust_2 = z_0_est + (z_0_est+st.norm.ppf(0.975)) / (1 - acc_factor*(z_0_est+st.norm.ppf(0.975)))
alpha_1 = st.norm.cdf(adjust_1)
alpha_2 = st.norm.cdf(adjust_2)

# quantile
np.quantile(result_bs, [0.025, 0.975])
# BCa
np.quantile(result_bs, [alpha_1, alpha_2])