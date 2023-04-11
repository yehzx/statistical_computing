import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import time 


# set seed = 1000
# np.random.seed(1000)
n = 100000

# =====
# Utils
# =====
def timer(iter):
    def wrap(func):
        def inner_func(*args, **kwargs):
            start = time.perf_counter()
            for i in range(iter):
                func(*args, **kwargs)
            end = time.perf_counter()
            print(f"Elapsed time: {round(end - start, 3)}sec")
        return inner_func
    return wrap


def g(x):
    return x**2 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

def weight_of_g(x):
    return x**2


def weight_of_g_filter(x):
    return np.where(x > 1, x**2, 0)


def gen_norm_var_via_inverse(n):
    uni_var = np.random.uniform(st.norm.cdf(1), 1, size=n)
    normal_var = st.norm.ppf(uni_var)
    return normal_var


def gen_norm_var_via_inverse_bin(n, bin):
    uni_range = np.linspace(st.norm.cdf(1), 1, bin + 1)
    for b in range(bin):
        uni_var = np.random.uniform(uni_range[b], uni_range[b+1], size=int(n/bin))
        normal_var = st.norm.ppf(uni_var)
        yield normal_var

# 1. Direct Monde Carlo
@timer(iter=1)
def direct_monte_carlo(n):
    total_sample = 0
    sample_li = []
    while total_sample < n:
        normal_var = np.random.normal(size=10000)
        normal_var = normal_var[normal_var > 1]
        total_sample += len(normal_var)
        sample_li.append(normal_var)
    sample_li = np.concatenate(sample_li)
    sample_li = sample_li[:n]

    # nomalization
    est_li = weight_of_g(sample_li)
    theta_est = np.mean(est_li) * (1 - st.norm.cdf(1))
    return theta_est

# 3. Variable transformation of the integrand to a interval of [0, 1]
def integrand_0_1(n):
    normal_var = gen_norm_var_via_inverse(n)

    # nomalization
    est_li = weight_of_g(normal_var) * (1 - st.norm.cdf(1))
    theta_est = np.mean(est_li)
    return theta_est


def gen_norm_var_via_inverse_bin(n, bin):
    uni_range = np.linspace(st.norm.cdf(1), 1, bin + 1)
    for b in range(bin):
        uni_var = np.random.uniform(uni_range[b], uni_range[b+1], size=int(n/bin))
        normal_var = st.norm.ppf(uni_var)
        yield normal_var

# 5. Stratified Sampling
def stratified_sampling(n, bin):
    normal_var_li = []
    bin = 5
    genarator = gen_norm_var_via_inverse_bin(n, bin)
    for b in range(bin):
        normal_var_li.append(next(genarator))
    
    est_li = np.array(list(map(weight_of_g, normal_var_li))) * (1 - st.norm.cdf(1)) / bin
    est_li = np.mean(est_li, axis=1)
    theta_est = np.sum(est_li)

    return theta_est


def cal_f1_result(n):
    x = np.random.normal(size=n)
    x = np.abs(x) +1
    density = g(x) / f1(x-1)
    theta_hat = np.mean(density)
    convergence_estimation = np.divide(np.cumsum(density), np.arange(1, n+1))
    variance = np.var(density)

    return theta_hat, convergence_estimation, variance

def single_control_variate(n):
    pass
    # uni_var = np.random.uniform(st.norm.cdf(1), 1, size=n)
    # x = gen_norm_var_via_inverse_given(uni_var)
    # est_li = weight_of_g(x) * (1 - st.norm.cdf(1))

    # control_1_li = uni_var
    # control_2_li = x_squared(uni_var)
    # control_2_mean, _ = integrate.quad(x_squared, st.norm.cdf(1), 1) / (1 - st.norm.cdf(1))

    # cov_f_h = np.cov(est_li, control_1_li, ddof=0)
    # c = - cov_f_h[1, 0] / cov_f_h[0, 0]
    # est_li = est_li + c * (control_2_li - control_2_mean)