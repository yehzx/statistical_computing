import numpy as np
import matplotlib.pyplot as plt
from final_main import (BirthWeightData,
                        CHAIN, SAMPLE,
                        plot_result_param,
                        run_plot_gr_method)
import time

np.seterr(all='raise')

# Q1. (use metropolis=hastings to sample sigma and tau)
np.random.seed(1000)
mh_result_1_li = []
time_mh_1_li = []
for i in range(CHAIN):
    start = time.perf_counter()
    mh_result_1_li.append(BirthWeightData(sample_size=SAMPLE, method="metropolis-hastings", joint_pdf="cauchy_cauchy"))
    end = time.perf_counter()
    time_mh_1_li.append(end-start)
print(f"time: {np.mean(time_mh_1_li)}, {np.sqrt(np.var(time_mh_1_li))}")

# Q2. AR sampling of sigma
np.random.seed(1000)
ar_result_li = []
time_ar_li = []
for i in range(CHAIN):
    start = time.perf_counter()
    ar_result_li.append(BirthWeightData(sample_size=SAMPLE, method="acceptance-rejection", joint_pdf="invgamma_cauchy"))
    end = time.perf_counter()
    time_ar_li.append(end-start)
print(f"time: {np.mean(time_ar_li)}, {np.var(time_ar_li)}")

# Q3. independent metropolis-hastings sampling of sigma
np.random.seed(1000)
imh_result_li = []
time_imh_li = []
for i in range(CHAIN):
    start = time.perf_counter()
    imh_result_li.append(BirthWeightData(sample_size=SAMPLE, method="independent metropolis-hastings", joint_pdf="invgamma_cauchy"))
    end = time.perf_counter()
    time_imh_li.append(end-start)
print(f"time: {np.mean(time_imh_li)}, {np.var(time_imh_li)}")


# Q4. metropolis-hastings sampling of sigma
np.random.seed(1000)
mh_result_2_li = []
time_mh_2_li = []
for i in range(CHAIN):
    start = time.perf_counter()
    mh_result_2_li.append(BirthWeightData(sample_size=SAMPLE, method="metropolis-hastings", joint_pdf="invgamma_cauchy"))
    end = time.perf_counter()
    time_mh_2_li.append(end-start)
print(f"time: {np.mean(time_mh_2_li)}, {np.var(time_mh_2_li)}")

plot_result_param(mh_result_1_li, type="distribution", single=False)
plot_result_param(ar_result_li, type="distribution", single=False)
plot_result_param(imh_result_li, type="distribution", single=False)
plot_result_param(mh_result_2_li, type="distribution", single=False)

plot_result_param(mh_result_1_li, type="trace", single=False)
plot_result_param(ar_result_li, type="trace", single=False)
plot_result_param(imh_result_li, type="trace", single=False)
plot_result_param(mh_result_2_li, type="trace", single=False)

run_plot_gr_method(mh_result_1_li)
run_plot_gr_method(ar_result_li)
run_plot_gr_method(imh_result_li)
run_plot_gr_method(mh_result_2_li)
