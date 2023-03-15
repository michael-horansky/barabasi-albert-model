
import numpy as np
import random as rd
import scipy as sp

import matplotlib.pyplot as plt
import time

def p_k(k, m):
    return(2.0 * m * (m + 1.0) / (k * (k + 1.0) * (k + 2.0)))

def expected_max_k(m, N, k_max = 1000000):
    # make sure k_max >> expected k_max (also preferably k_max >> number of steps you drive the network for)
    res_sum = 0.0
    number_of_repetitions = k_max - m
    start_time = time.time()
    progress_percentage = 0.0
    for k in range(m, k_max):
        if np.floor((k-m) / number_of_repetitions * 100) > progress_percentage:
            progress_percentage = np.floor((k-m) / number_of_repetitions * 100)
            print("Analysis in progress: " + str(progress_percentage) + "%; est. time of finish: " + time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress_percentage + start_time )), end='\r')
        p_k = np.power(1.0 - m * (m+1.0) / ((k + 1.0) * (k + 2.0)), N) - np.power(1.0 - m * (m+1.0) / ((k + 0.0) * (k + 1.0)), N)
        res_sum += k * p_k
    print("Analysis done.                                                     ") #this is SUCH an ugly solution.
    return(res_sum)


m = 3
k_ceil = int(1e6)

k_space = np.arange(m, k_ceil)
k_space_probabilities = p_k(k_space, m)

print("Sum of probabilities =", sum(k_space_probabilities))

N_max_space = [int(elem) for elem in [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5]]

k_max_avg = []
k_max_std = []
N_m = 100

for N_i in range(len(N_max_space)):
    
    print("Analyzing N_max =", N_max_space[N_i])
    cur_k_max_array = []
    
    for i in range(N_m):
        # We generate a sample
        k_sample = np.random.choice(k_space, N_max_space[N_i], replace=True, p=k_space_probabilities)
        cur_k_max_array.append(max(k_sample))
    k_max_avg.append(np.average(cur_k_max_array))
    k_max_std.append(np.std(cur_k_max_array))

theoretical_k_max = expected_max_k(m, N_max_space)

plt.errorbar(N_max_space, k_max_avg, yerr=k_max_std / np.sqrt(N_m), label='values')
plt.plot(N_max_space, theoretical_k_max, label='prediction')
plt.legend()
plt.show()
