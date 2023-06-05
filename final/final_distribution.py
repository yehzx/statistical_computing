import numpy as np
import scipy.stats as st
from scipy.special import gamma
import sys

def generate_inverse_gamma(a, b):
    return float(1 / np.random.gamma(shape=a, scale=b, size=1))

def cal_trun_cauchy_density(x, b):
    # assert b > 0, "invalid trun_cauchy parameter b"
    # return 2 / np.pi / (b * (1 + (x/b)**2)) if x >= 0 else 0
    return 2 / np.pi / (b * (1 + (x/b)**2))

def cal_inverse_gamma_density(x, a, b):
    try:
        return b**a * np.exp(-b / x) * x**(-a-1) / gamma(a)
    except (ZeroDivisionError, FloatingPointError):
        return 0
    except OverflowError:
        return sys.float_info.max

def cal_chi_2_density(target, param):
    try:
        return st.chi2.pdf(target, df=param)
    except FloatingPointError:
        return 0
def generate_chi_2(df):
    return np.random.chisquare(df)


# import matplotlib.pyplot as plt
# x = np.linspace(0.01, 5, 100)
# a = 2.5
# b = 2.5
# plt.plot(x, cal_inverse_gamma_density(x, a, b))