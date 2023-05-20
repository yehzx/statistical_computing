import time

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy.integrate as integrate
import scipy.stats as st

plt.style.use("science")
plt.rc("font", size=14)
def initial_plot():
    ax, fig = plt.subplots(figsize=(6.4, 4.8))
    plt.xlabel("Iteration")
    plt.ylabel("Estimation of Integral")
    return ax, fig

# set seed = 1000
np.random.seed(1000)
n = 10000

# =====
# Utils
# =====
def timer(iter):
    def wrap(func):
        def inner_func(*args, **kwargs):
            start = time.perf_counter()
            for i in range(iter):
                result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            print(f"Elapsed time: {round(elapsed, 3)}ms")
            return result
        return inner_func
    return wrap

# def cal_variance(iter=100):
#     def wrap(func):
#         def inner_func(*args, **kwargs):
#             result = []
#             for i in range(iter):
#                 try:
#                     theta_est, variance, conv_est = func(*args, **kwargs)
#                 except ValueError:
#                     theta_est, conv_est = func(*args, **kwargs)  
#                 result.append(theta_est)
#             return np.var
#         return inner_func
#     return wrap

def cal_variance(iter, func, *args, **kwargs):
    result = []
    for i in range(iter):
        try:
            theta_est, variance, conv_est = func(*args, **kwargs)
        except ValueError:
            theta_est, conv_est = func(*args, **kwargs)  
        result.append(theta_est)
        
    return np.var(result)


def plot_convergence(label, iter=100):
    def wrap(func):
        def inner_func(*args, **kwargs):
            np.random.seed(1000)
            try:
                theta_est, variance, conv_est = func(*args, **kwargs)
            except ValueError:
                theta_est, conv_est = func(*args, **kwargs)

            n = kwargs["n"]
            print(f"{label}")
            print(f"Estimation from {n} samples: {theta_est}")
            print(f"Variance of {iter} estimations:{cal_variance(iter, func, *args, **kwargs)}")
            plt.clf()
            ax, fig = initial_plot()
            try:
                fig.plot(np.arange(1, n+1), conv_est, color="b", label=label)
            except ValueError:
                bin = kwargs["bin"]
                fig.plot(np.arange(1, n/bin+1), conv_est, color="b", label=label)
            plt.show()
        return inner_func
    return wrap


# target function
def g(x):
    return x**2 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

def weight_of_g(x):
    return x**2


# 1. Direct Monte Carlo
def weight_of_g_filter(x):
    return np.where(x > 1, x**2, 0)

@plot_convergence(label="1_DMC", iter=100)
def direct_monte_carlo(n):
    normal_var = np.random.normal(size=n)
    est_li = weight_of_g_filter(normal_var)
    theta_est = np.mean(est_li)

    # this is the variance within each estimation
    # true variance for multiple estimations is obtained from 100 iters (not shown here)
    # all "variance" of each method below has the same implication.
    variance = np.var(est_li)
    conv_est = np.divide(np.cumsum(est_li), np.arange(1, n+1))

    return theta_est, variance, conv_est


# 2. Importance Sampling
def normal_dist_2(x):
    return 2 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

def normalized_exp(x):
    return np.exp(-x)

def normal_dist(x):
    return 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

@plot_convergence(label="2_IS", iter=100)
def importance_sampling(n):
    x = np.random.normal(size=n)
    x = np.abs(x) + 1

    density = g(x) / normal_dist_2(x-1)
    theta_est = np.mean(density)
    conv_est = np.divide(np.cumsum(density), np.arange(1, n+1))
    variance = np.var(density)

    return theta_est, variance, conv_est


@plot_convergence(label="2_2_SNIS", iter=100)
def self_normalized_importance_sampling(n):
    x = np.random.exponential(size=n) + 1
    w = normal_dist(x) / normalized_exp(x-1)
    density = x**2 * w
    theta_est = np.mean(density) / np.mean(w) * (1 - st.norm.cdf(1))
    variance = np.var(density)
    conv_est = np.divide(np.cumsum(density), np.arange(1, n+1))

    return theta_est, variance, conv_est


# 3. Variable transformation of the integrand to a interval of [0, 1]
def gen_norm_var_via_inverse(start, end):
    uni_var = np.random.uniform(start, end, size=n)
    normal_var = st.norm.ppf(uni_var)
    return normal_var

@plot_convergence(label="3_VT", iter=100)
def integrand_0_1(n):
    normal_var = gen_norm_var_via_inverse(st.norm.cdf(1), 1)
    est_li = weight_of_g(normal_var) * (1 - st.norm.cdf(1))
    theta_est = np.mean(est_li)
    variance = np.var(est_li)
    conv_est = np.divide(np.cumsum(est_li), np.arange(1, n+1))

    return theta_est, variance, conv_est


# 4. Two control variate
def gen_norm_var_via_inverse_given(uni_var):
    normal_var = st.norm.ppf(uni_var)
    return normal_var

def x_squared(x):
    return x**2

@plot_convergence(label="4_CV", iter=100)
def two_control_variate(n):
    # variable transformation
    uni_var = np.random.uniform(st.norm.cdf(1), 1, size=n)
    x = gen_norm_var_via_inverse_given(uni_var)
    est_li = weight_of_g(x) * (1 - st.norm.cdf(1))

    # control_1: uni_var
    control_1_li = uni_var
    
    # control_2: x ** 2
    control_2_li = x_squared(uni_var)
    control_2_mean, _ = integrate.quad(x_squared, st.norm.cdf(1), 1) / (1 - st.norm.cdf(1))

    # solve best c_1, c_2
    cov_f_h = np.cov(est_li, control_1_li, ddof=0)
    cov_f_g = np.cov(est_li, control_2_li, ddof=0)
    cov_h_g = np.cov(control_1_li, control_2_li, ddof=0)

    A = np.array([[cov_h_g[0, 0], cov_h_g[1, 0]], [cov_h_g[1, 0], cov_h_g[1, 1]]])
    B = np.array([-cov_f_h[1, 0], -cov_f_g[1, 0]])
    c_1, c_2 = np.linalg.solve(A, B)

    est_li = est_li + c_1 * (control_1_li - (st.norm.cdf(1)+1) / 2) + \
             c_2 * (control_2_li - control_2_mean)
    
    theta_est = np.mean(est_li)
    variance = np.var(est_li)
    conv_est = np.divide(np.cumsum(est_li), np.arange(1, n+1))

    return theta_est, variance, conv_est


# 5. Stratified Sampling
def gen_norm_var_via_inverse_bin(n, bin):
    uni_range = np.linspace(st.norm.cdf(1), 1, bin+1)
    normal_var_li = []
    for b in range(bin):
        uni_var = np.random.uniform(uni_range[b], uni_range[b+1], size=int(n/bin))
        normal_var = st.norm.ppf(uni_var)
        normal_var_li.append(normal_var)
    return normal_var_li

@plot_convergence(label="5_SS", iter=100)
def stratified_sampling(n, bin):
    normal_var_li = gen_norm_var_via_inverse_bin(n, bin)
    est_li = list(map(weight_of_g, normal_var_li))
    bin_den_norm = (1 - st.norm.cdf(1)) / bin
    est_li = np.array(est_li) * bin_den_norm

    stratified_cumsum = list(map(np.cumsum, est_li))
    est_li_for_conv_est = np.sum(stratified_cumsum, axis=0)
    to_range = int(n/bin) + 1
    conv_est = np.divide(est_li_for_conv_est, np.arange(1, to_range))
    
    est_li = np.mean(est_li, axis=1)
    theta_est = np.sum(est_li)

    return theta_est, conv_est

# plot different bin numbers
# var_li = []
# for i in range(1, 101):
#     est_li = []
#     for j in range(100):
#         est = stratified_sampling(n, i)[0]
#         est_li.append(est)
#     var_li.append(np.var(est_li))
# var_li
# plt.clf()
# ax, fig = initial_plot()
# plt.xlabel("Bin")
# plt.ylabel("Variance")
# plt.plot(np.arange(2, 101), var_li[1:], color="b")
# var_li[49]
# plt.show()


# 6. tile density as importance function
def mgf_normal(u, sigma, t):
    return np.exp(u*t + (sigma**2) * (t**2) / 2)

def normal_dist(x):
    return 1 / np.sqrt(2*np.pi) * np.exp(-x**2 / 2)

def tile_function(x, t):
    return np.exp(x*t) * normal_dist(x) / mgf_normal(0, 1, t)

def cal_density(x, t):
    return np.where(x > 1, g(x) / tile_function(x, t), 0)

@plot_convergence(label="6_TIS", iter=100)
def tile_importance_sampling(n, t):
    # ver.1
    # x = np.random.normal(size=n) + t 
    # density = cal_density(x, t)

    # ver.2
    x = np.random.normal(size=n)
    x = np.abs(x) + t

    density = g(x - t + 1) / (2 * tile_function(x, t))

    theta_est = np.mean(density)
    conv_est = np.divide(np.cumsum(density), np.arange(1, n+1))
    variance = np.var(density)

    return theta_est, variance, conv_est


direct_monte_carlo(n=n)
importance_sampling(n=n)
self_normalized_importance_sampling(n=n)
integrand_0_1(n=n)
two_control_variate(n=n)
stratified_sampling(n=n, bin=5)
tile_importance_sampling(n=n, t=3)