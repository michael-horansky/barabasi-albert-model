from BA_network_class import *


def loglog_fit(xspace, yspace, yerr = None):
    x_log = np.log(xspace)
    y_log = np.log(yspace)
    if yerr == None:
        fit_vals, fit_cov = sp.optimize.curve_fit(lambda x, a, b : a * x + b, x_log, y_log)
    else:
        yerr_log = np.log(yerr)
        fit_vals, fit_cov = sp.optimize.curve_fit(lambda x, a, b : a * x + b, x_log, y_log, sigma=yerr_log)
    res_k = fit_vals[0]
    std_k = np.sqrt(fit_cov[0][0])
    res_b = fit_vals[1]
    std_b = np.sqrt(fit_cov[1][1])
    return(res_k, std_k, res_b, std_b, x_log.min(), x_log.max())

def get_b_from_line(xspace, yspace, k, yerr = None):
    x_log = np.log(xspace)
    y_log = np.log(yspace)
    if yerr == None:
        fit_vals, fit_cov = sp.optimize.curve_fit(lambda x, b : k * x + b, x_log, y_log)
    else:
        yerr_log = np.log(yerr) #TODO THIS IS WRONG - USE DERIVATIVES
        fit_vals, fit_cov = sp.optimize.curve_fit(lambda x, b : k * x + b, x_log, y_log, sigma=yerr_log)
    res_b = fit_vals[0]
    std_b = np.sqrt(fit_cov[0][0])
    return(res_b, std_b, x_log.min(), x_log.max())

def m_scaling_test(m_space, method, N_m=1, PS = 'PA', *args):
    # m_space = array of m values to test on
    # method = method of the BA network class that returns a measurement value
    # N_m = number of repetitions per m value datapoint
    # *args = arguments to pass to BA_network.method
    res_avg_array = []
    if N_m > 1:
        res_std_array = []
    print(f"--------- Commencing testing on {len(m_space)} instances --------")
    for m_val in m_space:
        print(f"Current m = {m_val}")
        
        res_matrix = []
        for i in range(N_m):
            print("  current index =", i+1)
            cur_BA_network = BA_network(m = m_val, probability_strategy = PS)
            res_matrix.append(method(cur_BA_network, *args))
        if N_m > 1 :
            res_avg_array.append(np.average(res_matrix, axis=0))
            res_std_array.append(np.std(res_matrix, axis=0))
        else:
            res_avg_array.append(res_matrix[0])
        #print(f"  Result = {res_avg_array[-1]}")
    if N_m > 1:
        return(res_avg_array, res_std_array)
    return(res_avg_array)

# special scaling test for degree distribution that handles the statistical data churning on the ugly logbinned arrays
def m_scaling_degree_distribution(dataset_name = 'dataset', m_space = [3], N_m=1, PS = 'PA', N_max = 1e4, bin_scale = 1.3):
    # m_space = array of m values to test on
    # N_m = number of repetitions per m value datapoint
    # PS is the strategy
    # N_max = final number of nodes - THIS CAN BE A LIST
    if type(N_max) != list:
        N_max = [N_max] * len(m_space)
    res_avg_array = []
    print(f"--------- Commencing testing on {len(m_space)} instances (dataset '{dataset_name}') --------")
    for m_i in range(len(m_space)):
        m_val = m_space[m_i]
        print(f"Current m = {m_val}; N_max = {N_max[m_i]}")
        
        x_matrix = []
        y_PDF_matrix = []
        y_counts_matrix = []
        binedges_matrix = []
        widest_sample_index = -1
        x_max = 0
        for i in range(N_m):
            print("  current index =", i+1)
            cur_BA_network = BA_network(m = m_val, probability_strategy = PS)
            
            degree_array = cur_BA_network.get_degree_distribution(N_max = N_max[m_i])
            x, y_PDF, y_counts, binedges = logbin(degree_array, scale = bin_scale, x_min = m_val)
            
            x_matrix.append(x)
            y_PDF_matrix.append(y_PDF)
            y_counts_matrix.append(y_counts)
            binedges_matrix.append(binedges)
            
            if max(x) > x_max:
                x_max = max(x)
                widest_sample_index = i
        
        if N_m == 1:
            res_avg_array.append([x_matrix[0].copy(),y_PDF_matrix[0], np.zeros(len(y_PDF_matrix[0])),y_counts_matrix[0], np.zeros(len(y_counts_matrix[0])),binedges_matrix[0]])
        else:
            # calculate the ugly statisics
            
            # first, locate the x-array and binedges-array that spans the entire dataset
            x = x_matrix[widest_sample_index]
            binedges = binedges_matrix[widest_sample_index]
            print("Length of sample-spanning x-array =", len(x))
            # now append empty bins to all lacking y-arrays - assume the non-spanning binedges are always left-aligned
            for i in range(N_m):
                length_difference = len(x) - len(x_matrix[i])
                #print("kuk", length_difference)
                if length_difference > 0:
                    y_PDF_matrix[i] = np.concatenate((y_PDF_matrix[i], np.zeros(length_difference)))
                    y_counts_matrix[i] = np.concatenate((y_counts_matrix[i], np.zeros(length_difference)))
                #print("lalaho", )
            # now calculate the statistics
            y_PDF_avg = np.average(y_PDF_matrix, axis=0)
            y_PDF_std = np.std(y_PDF_matrix, axis=0)
            y_counts_avg = np.average(y_counts_matrix, axis=0)
            y_counts_std = np.std(y_counts_matrix, axis=0)
            res_avg_array.append([x,y_PDF_avg,y_PDF_std,y_counts_avg,y_counts_std,binedges])
        """if N_m > 1 :
            res_avg_array.append(np.average(res_matrix, axis=0))
            res_std_array.append(np.std(res_matrix, axis=0))
        else:
            res_avg_array.append(res_matrix[0])"""
        #print(f"  Result = {res_avg_array[-1]}")
    
    # We now save the res_avg_array into a txt file
    """
    The file has the form:
        1 [PS, N_m, bin_scale]
        2 [m1, m2, m3...]
        3 [N_max1, N_max2, N_max3...]
        Then for each m_val, there are 6 rows:
            1 [x]
            2 [y_PDF]
            3 [y_PDF_std]
            4 [y_counts]
            5 [y_counts_std]
            5 [binedges]
    """
    output_delim = ', '
    m_scaling_degree_distribution_output_file = open("data/m_k_scaling_" + dataset_name + ".txt", mode="w")
    #filename_list_stringed = 'L'.join([str(elem) for elem in my_filename_list])
    #meta_config_file.write(filename_list_stringed)
    m_scaling_degree_distribution_output_file.write(f'{PS}, {N_m}, {bin_scale}\n')
    m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in m_space]) + '\n')
    m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in N_max]) + '\n')
    for m_i in range(len(m_space)):
        for res_i in range(len(res_avg_array[m_i])):
            m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in res_avg_array[m_i][res_i]]) + '\n')
    m_scaling_degree_distribution_output_file.close()
    
    
    return(res_avg_array)

def load_m_scaling_degree_distribution(dataset_name, keep_descriptors = False):
    
    # Input file configuration parameters
    head_line_number = 3
    line_number_per_m_val = 6
    data_line_types = [float, float, float, float, float, int]
    
    m_scaling_degree_distribution_input_file = open("data/m_k_scaling_" + dataset_name + ".txt", mode="r")
    input_lines = m_scaling_degree_distribution_input_file.readlines()
    m_scaling_degree_distribution_input_file.close()
    
    head_lines = [item.split(", ") for item in input_lines[:head_line_number]]
    #data_lines = [[float(elem) for elem in item.split(", ")] for item in input_lines[head_line_number:]]
    data_lines = [item.split(", ") for item in input_lines[head_line_number:]]
    
    PS = head_lines[0][0]
    N_m = int(head_lines[0][1])
    bin_scale = float(head_lines[0][2])
    
    m_space = [int(elem) for elem in head_lines[1]]
    N_max_space = [float(elem) for elem in head_lines[2]]
    
    
    res_array = []
    try:
        for m_i in range(len(m_space)):
            res_array.append([])
            for dataline_i in range(line_number_per_m_val):
                res_array[m_i].append(np.array([data_line_types[dataline_i](item) for item in data_lines[m_i * line_number_per_m_val + dataline_i]]))
    except IndexError:
        print("ERROR: Input file broken")
        return(-1)
        
    
    print(f"--------------- Loaded dataset '{dataset_name}' --------------")
    print(f"Probability strategy: {PS}; Number of repetitions per datapoint = {N_m}; Bin scale = {bin_scale}")
    print(f"m_space = {m_space}; N_max_space = {N_max_space}")
    if keep_descriptors:
        return(PS, N_m, bin_scale, m_space, N_max_space, res_array)
    return(res_array)
    
def N_scaling_degree_distribution(dataset_name = 'dataset', N_space = [1e4], N_m=1, m = 3, PS = 'PA', bin_scale = 1.3):
    # N_space = array of N_max values to test on
    # N_m = number of repetitions per m value datapoint
    if type(m) != list:
        m = [m] * len(N_space)
    res_avg_array = []
    k_max_avg_array = []
    k_max_std_array = []
    print(f"--------- Commencing testing on {len(N_space)} instances (dataset '{dataset_name}') --------")
    for N_i in range(len(N_space)):
        N_max_val = N_space[N_i]
        print(f"Current N_max = {N_max_val}; m = {m[N_i]}")
        
        x_matrix = []
        y_PDF_matrix = []
        y_counts_matrix = []
        binedges_matrix = []
        widest_sample_index = -1
        x_max = 0
        cur_k_max_array = []
        for i in range(N_m):
            print("  current index =", i+1)
            cur_BA_network = BA_network(m = m[N_i], probability_strategy = PS)
            
            degree_array = cur_BA_network.get_degree_distribution(N_max = N_space[N_i])
            cur_k_max_array.append(max(degree_array))
            x, y_PDF, y_counts, binedges = logbin(degree_array, scale = bin_scale, x_min = m[N_i])
            
            x_matrix.append(x)
            y_PDF_matrix.append(y_PDF)
            y_counts_matrix.append(y_counts)
            binedges_matrix.append(binedges)
            
            if max(x) > x_max:
                x_max = max(x)
                widest_sample_index = i
        
        if N_m == 1:
            res_avg_array.append([x_matrix[0].copy(),y_PDF_matrix[0], np.zeros(len(y_PDF_matrix[0])),y_counts_matrix[0], np.zeros(len(y_counts_matrix[0])),binedges_matrix[0]])
            k_max_avg_array.append(cur_k_max_array[0])
            k_max_std_array.append(0.0)
        else:
            # calculate the ugly statisics
            
            # first, locate the x-array and binedges-array that spans the entire dataset
            x = x_matrix[widest_sample_index]
            binedges = binedges_matrix[widest_sample_index]
            print("Length of sample-spanning x-array =", len(x))
            # now append empty bins to all lacking y-arrays - assume the non-spanning binedges are always left-aligned
            for i in range(N_m):
                length_difference = len(x) - len(x_matrix[i])
                if length_difference > 0:
                    y_PDF_matrix[i] = np.concatenate((y_PDF_matrix[i], np.zeros(length_difference)))
                    y_counts_matrix[i] = np.concatenate((y_counts_matrix[i], np.zeros(length_difference)))
            # now calculate the statistics
            y_PDF_avg = np.average(y_PDF_matrix, axis=0)
            y_PDF_std = np.std(y_PDF_matrix, axis=0)
            y_counts_avg = np.average(y_counts_matrix, axis=0)
            y_counts_std = np.std(y_counts_matrix, axis=0)
            res_avg_array.append([x,y_PDF_avg,y_PDF_std,y_counts_avg,y_counts_std,binedges])
            k_max_avg_array.append(np.average(cur_k_max_array))
            k_max_std_array.append(np.std(cur_k_max_array))
    
    # We now save the res_avg_array into a txt file
    """
    The file has the form:
        1 [PS, N_m, bin_scale]
        2 [m1, m2, m3...]
        3 [N_max1, N_max2, N_max3...]
        4 [k_max1, k_max2, k_max3...]
        5 [k_max_std1, k_max_std2, k_max_std3...]
        Then for each m_val, there are 6 rows:
            1 [x]
            2 [y_PDF]
            3 [y_PDF_std]
            4 [y_counts]
            5 [y_counts_std]
            5 [binedges]
    """
    output_delim = ', '
    N_scaling_degree_distribution_output_file = open("data/N_k_scaling_" + dataset_name + ".txt", mode="w")
    #filename_list_stringed = 'L'.join([str(elem) for elem in my_filename_list])
    #meta_config_file.write(filename_list_stringed)
    N_scaling_degree_distribution_output_file.write(f'{PS}, {N_m}, {bin_scale}\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in m]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in N_space]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in k_max_avg_array]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in k_max_std_array]) + '\n')
    for N_i in range(len(N_space)):
        for res_i in range(len(res_avg_array[N_i])):
            N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in res_avg_array[N_i][res_i]]) + '\n')
    N_scaling_degree_distribution_output_file.close()
    
    return(k_max_avg_array, k_max_std_array,res_avg_array)

def load_N_scaling_degree_distribution(dataset_name, keep_descriptors = False):
    
    # Input file configuration parameters
    head_line_number = 5
    line_number_per_m_val = 6
    data_line_types = [float, float, float, float, float, int]
    
    N_scaling_degree_distribution_input_file = open("data/N_k_scaling_" + dataset_name + ".txt", mode="r")
    input_lines = N_scaling_degree_distribution_input_file.readlines()
    N_scaling_degree_distribution_input_file.close()
    
    head_lines = [item.split(", ") for item in input_lines[:head_line_number]]
    #data_lines = [[float(elem) for elem in item.split(", ")] for item in input_lines[head_line_number:]]
    data_lines = [item.split(", ") for item in input_lines[head_line_number:]]
    
    PS = head_lines[0][0]
    N_m = int(head_lines[0][1])
    bin_scale = float(head_lines[0][2])
    
    m_space = [int(elem) for elem in head_lines[1]]
    N_max_space = [float(elem) for elem in head_lines[2]]
    
    k_max_avg_array = [float(elem) for elem in head_lines[3]]
    k_max_std_array = [float(elem) for elem in head_lines[4]]
    
    
    res_array = []
    try:
        for m_i in range(len(m_space)):
            res_array.append([])
            for dataline_i in range(line_number_per_m_val):
                res_array[m_i].append(np.array([data_line_types[dataline_i](item) for item in data_lines[m_i * line_number_per_m_val + dataline_i]]))
    except IndexError:
        print("ERROR: Input file broken")
        return(-1)
        
    
    print(f"--------------- Loaded dataset '{dataset_name}' --------------")
    print(f"Probability strategy: {PS}; Number of repetitions per datapoint = {N_m}; Bin scale = {bin_scale}")
    print(f"m_space = {m_space}; N_max_space = {N_max_space}")
    if keep_descriptors:
        return(PS, N_m, bin_scale, m_space, N_max_space, k_max_avg_array, k_max_std_array, res_array)
    return(k_max_avg_array, k_max_std_array, res_array)

def N_scaling_test(N_space, method, N_m=1, m = 3, PS = 'PA', *args):
    # N_space = array of N_max values to test on
    # method = method of the BA network class that returns a measurement value
    # N_m = number of repetitions per m value datapoint
    # *args = arguments to pass to BA_network.method (first argument MUST BE N_max)
    res_avg_array = []
    if N_m > 1:
        res_std_array = []
    print(f"--------- Commencing testing on {len(N_space)} instances --------")
    for N_val in N_space:
        print(f"Current N_max = {N_val}")
        
        res_matrix = []
        for i in range(N_m):
            print("  current index =", i+1)
            cur_BA_network = BA_network(m = m, probability_strategy = PS)
            res_matrix.append(method(cur_BA_network, N_val,  *args))
        if N_m > 1 :
            res_avg_array.append(np.average(res_matrix, axis=0))
            res_std_array.append(np.std(res_matrix, axis=0))
        else:
            res_avg_array.append(res_matrix[0])
        #print(f"  Result = {res_avg_array[-1]}")
    if N_m > 1:
        return(res_avg_array, res_std_array)
    return(res_avg_array)

def theoretical_gamma_p_infty(k, m):
    return((2.0 * m * (m + 1.0)) / (k * (k + 1.0) * (k + 2.0)))

def primitive_function_gamma_p_infty(k, m):
    # This is the promitive function of theoretical_gamma_p_infty(k, m)
    return(m * (m + 1.0) * (np.log(k) - 2.0 * np.log(k + 1.0) + np.log(k + 2.0)))

def expected_frequency_p_infty(k, m):
    # this is a discrete version of theoretical_gamma_p_infty
    if k == m:
        return(2.0 / (m + 2.0))
    else:
        return((2.0 * m * (m + 1.0)) / (k * (k + 1.0) * (k + 2.0)))

def expected_bin_frequency_p_infty(m, k_min, k_max = -1):
    # counts the theoretical frequency for k_min <= k < k_max
    # if k_max == -1, it is interpreted as infinity
    if k_max == -1:
        return(m * (m + 1.0) / (k_min * (k_min + 1.0)))
    return(m * (m + 1.0) * (k_max * (k_max + 1.0) - k_min * (k_min + 1.0)) / (k_max * (k_max + 1.0) * k_min * (k_min + 1.0)))
def theoretical_binned_gamma_p_infty(bin_edges, m):
    # This calculates the expected normalized AND unnormalized binned distribution of the gamma func. p_infty function
    # y[i] = (bin_edges[i+1] - bin_edges[i])^-1 * \int_{bin_edges[i]}^{bin_edges[i+1]} f(k) dk
    y_PDF = (primitive_function_gamma_p_infty(bin_edges[1:], m) - primitive_function_gamma_p_infty(bin_edges[:-1], m)) / (bin_edges[1:] - bin_edges[:-1])
    y_counts = np.zeros(len(bin_edges)-1)
    for i in range(len(bin_edges)-1):
        y_counts[i] = expected_bin_frequency_p_infty(m, bin_edges[i], bin_edges[i+1])
    """for i in range(len(bin_edges)-2):
        y_counts[i] = expected_bin_frequency_p_infty(m, bin_edges[i], bin_edges[i+1])
    y_counts[-1] = expected_bin_frequency_p_infty(m, bin_edges[-2], -1)"""
    return(y_PDF, y_counts)
    # note that neither of these is normalized - the PDF is non-negligible even for bins outside of the bin_edges scope

def weighted_chi_sq(y_measurement, y_prediction, y_std):  
      chi_sq = np.sum( ((y_measurement-y_prediction)/y_std)**2/y_prediction )  
      return(chi_sq)

def sanitize_fat_tail(bincounts, binedges, theoretical_binned_frequency, m_val, min_bin_count = 5):
    print("  Sanitizing the fat-tail bins by combining them from the right until the minimum value isn't smaller than", min_bin_count)
    cum_bin_count = 0
    cutoff_index = len(bincounts)
    if sum(bincounts) < min_bin_count:
        print("  ERROR: Total count smaller than minimal bin count per bin")
        return(-1)
    while(min(bincounts[:cutoff_index]) < min_bin_count or cum_bin_count < min_bin_count):
        cutoff_index -= 1
        cum_bin_count += bincounts[cutoff_index]
    if cum_bin_count == 0:
        print("  LOG: All bins satisfied the min_bin_count condition already; no sanitization occurred")
        return(bincounts, binedges, np.concatenate((theoretical_binned_frequency[:-1], [theoretical_binned_frequency[-1] + expected_bin_frequency_p_infty(m_val, max(binedges))])))
    new_bincounts = np.concatenate((bincounts[:cutoff_index], [cum_bin_count]))
    new_binedges = binedges.copy()[:cutoff_index+1] #this implies an 'infinity' as a final entry, which we don't include, ofc
    # NOTE the goodness-of-fit doesn't actually take into account the x-position of the stuff. That's okay i guess. Well it does in the sense the theoretical prediction is an x sum
    new_theoretical_binned_frequency = np.concatenate((theoretical_binned_frequency[:cutoff_index], [expected_bin_frequency_p_infty(m_val,new_binedges[-1])]))
    print(f'  LOG: {len(bincounts) - cutoff_index} bins in the fat tail combined; new minimum bincount is {int(min(new_bincounts))}')
    return(new_bincounts, new_binedges, new_theoretical_binned_frequency)

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

def task1_3(new_dataset_name, m_space = [1, 3, 5], N_max = [5e4, 1e5, 2e5], N_m = 1):
    
    if type(N_max) != list:
        N_max = [N_max] * len(m_space)
    res_array = m_scaling_degree_distribution(new_dataset_name, m_space, N_m=N_m, PS = 'PA', N_max = N_max, bin_scale = 1.3)
    task1_3_analysis(N_m, m_space, N_max, res_array)

def task1_3_load(dataset_name):
    
    PS, N_m, bin_scale, m_space, N_max_space, res_array = load_m_scaling_degree_distribution(dataset_name, keep_descriptors = True)
    task1_3_analysis(N_m, m_space, N_max_space, res_array)

def task1_3_analysis(N_m, m_space, N_max, res_array):
    
    # res_array obtained either by m_scaling_degree_distribution or by load_m_scaling_degree_distribution

    x_list = []
    y_PDF_list = []
    y_counts_list = []
    y_PDF_err_list = []
    y_counts_err_list = []
    binedges_list = []
    theoretical_binned_distribution_list = []
    theoretical_binned_frequency_list = []
    
    for m_i in range(len(m_space)):
        print(f"---------------- Statistical analysis of m = {m_space[m_i]} ----------------")
        x, y_PDF, y_PDF_err, y_counts, y_counts_err, binedges = res_array[m_i]
        
        
        print("  Number of bins =", len(x))
        
        # y * (binedges[1:] - binedges[:-1]) sums up to 1 -> this is a good categorical PDF
        # USE BINEDGES, NOT X
        
        
        sum_y = 0
        for i in range(len(binedges)-1):
            sum_y += y_PDF[i] * (binedges[i+1] - binedges[i])
        print("  int(y_PDF) dk =", sum_y)
        sum_y = 0
        for i in range(len(binedges)-1):
            sum_y += y_counts[i]
        print("  sum(y_counts) over all bins (NOT values of k) =", sum_y)
        
        
        
        
        # Pearson's chi-squared test
        theoretical_binned_distribution, theoretical_binned_frequency = theoretical_binned_gamma_p_infty(binedges, m_space[m_i])
        print("  Sum of theoretical binned frequency counts =", sum(theoretical_binned_frequency))
        
        
        # theoretical distribution is normalized on m <= k <= infty, but in the measurement we have the N_max limitation, which
        # omits these high values. Hence we will extend the last bin in our counts (theoretical and measured), which has borders at
        # binedges[-2] and infinity. (basically setting binedges[-1] to infinity)
        
        #theoretical_binned_frequency_with_infinity = np.concatenate((theoretical_binned_frequency[:-1], [theoretical_binned_frequency[-1] + expected_bin_frequency_p_infty(m_space[m_i], max(binedges))]))
        #y_counts_with_infinity = y_counts.copy()#np.concatenate((y_counts, [0.0]))
        #NOTE we now do the "cumulative infinity binning"
        y_counts_with_infinity, binedges_with_infinity, theoretical_binned_frequency_with_infinity = sanitize_fat_tail(y_counts, binedges, theoretical_binned_frequency, m_space[m_i])
        print("  sum of theoretical binned frequency INCLUDING INFINITY =", sum(theoretical_binned_frequency_with_infinity))
        
        # BUT pearson is invalid for bins with counts smaller than 5, so we cannot include the infinity bin
        
        # chi-squared is run on counts, not frequencies -> we multiply both arrays by the total number of events = N_max
        
        chisq, p_val = sp.stats.chisquare(y_counts_with_infinity, theoretical_binned_frequency_with_infinity * N_max[m_i]) # this automatically sets df = len(list) - 1
        # p_val is the probability of obtaining a chi^2 exceeding the one obtained with H_0 being true.
        # it is effectively the probability that H_0 is true, given our measurement.
        
        # we also want to try out the WEIGHTED chi-squared test, where the summant is divided by the sample mean error squared
        # sample mean error = sample deviation / sqrt(sample size)
        
        chi_sq_df = len(y_counts_with_infinity)-1
        #w_chisq = weighted_chi_sq(y_counts_with_infinity, theoretical_binned_frequency_with_infinity * N_max[m_i], y_counts_err[m_i] / np.sqrt(N_m))
        print(f"  Pearson's chi-squared goodness-of-fit test for m = {m_space[m_i]} concluded with chi-squared = {chisq}, p = {p_val}")
        print(f"  Therefore the measured node degree distribution doesn't contradict the theoretical prediction with the probability P = {sp.stats.distributions.chi2.sf(chisq, chi_sq_df)}")
        #print(f"  weighted chi sq = {w_chisq}, p_val = {sp.stats.distributions.chi2.sf(w_chisq, chi_sq_df)}")
        #print(f"  reduced chi-squared = {w_chisq / chi_sq_df}")
        
        # save your stuff for plotting
        x_list.append(x)
        y_PDF_list.append(y_PDF)
        y_PDF_err_list.append(y_PDF_err)
        y_counts_list.append(y_counts)
        y_counts_err_list.append(y_counts_err)
        binedges_list.append(binedges)
        theoretical_binned_distribution_list.append(theoretical_binned_distribution)
        theoretical_binned_frequency_list.append(theoretical_binned_frequency)
    
    plt.subplot(2, 1, 1)
    plt.title("Probability distribution of node degree")
    
    plt.xlabel("$k$")
    plt.ylabel("$p_{\\infty}(k)$")
    for m_i in range(len(m_space)):
        x_nonzero = x_list[m_i][y_PDF_list[m_i]!=0]
        y_nonzero = y_PDF_list[m_i][y_PDF_list[m_i]!=0]
        
        yerr_nonzero = y_PDF_err_list[m_i][y_PDF_list[m_i]!=0]
        
        xerr_left = x_list[m_i] - binedges_list[m_i][:-1]
        xerr_right = binedges_list[m_i][1:] - x_list[m_i]
        
        xerr_left_nonzero = xerr_left[y_PDF_list[m_i]!=0]
        xerr_right_nonzero = xerr_right[y_PDF_list[m_i]!=0]
        
        if sum(yerr_nonzero) > 0:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], yerr = yerr_nonzero, fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        else:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        plt.loglog(x_list[m_i], theoretical_binned_distribution_list[m_i], linestyle='dotted', label=f'prediction ($m = {m_space[m_i]}$)')
    
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Bin count of node degree")
    
    plt.xlabel("k")
    plt.ylabel("$N_k$(bin)")
    
    for m_i in range(len(m_space)):
        x_nonzero = x_list[m_i][y_PDF_list[m_i]!=0]
        y_nonzero = y_counts_list[m_i][y_PDF_list[m_i]!=0]
        
        yerr_nonzero = y_counts_err_list[m_i][y_PDF_list[m_i]!=0]
        
        xerr_left = x_list[m_i] - binedges_list[m_i][:-1]
        xerr_right = binedges_list[m_i][1:] - x_list[m_i]
        
        xerr_left_nonzero = xerr_left[y_PDF_list[m_i]!=0]
        xerr_right_nonzero = xerr_right[y_PDF_list[m_i]!=0]
        
        if sum(yerr_nonzero) > 0:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], yerr = yerr_nonzero, fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        else:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        plt.loglog(x_list[m_i], theoretical_binned_frequency_list[m_i] * N_max[m_i], linestyle='dotted', label=f'prediction ($m = {m_space[m_i]}$)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def task1_4_expected_k_max(m_space_og = [1, 3, 5], N_space_og = [1e2, 1e3, 1e4, 1e5]):
    
    N_space = np.power(2.0, np.arange(20))#np.array([5, 1e1, 1e2, 1e3, 1e4, 1e5])
    m_space = np.arange(3, 15, 2).astype('int')
    
    plt.subplot(2, 1, 1)
    plt.title('Theoretical expected maximum degree for preferential attachement (N dependence)')
    plt.xlabel('Sample size $N$')
    plt.ylabel('$\\langle k_{{max}}\\rangle (N)$')
    
    for m_val in m_space_og:
        print("Analysing m =", m_val)
        mean_k_max_space = expected_max_k(m_val, N_space)
        plt.loglog(N_space, mean_k_max_space, 'x-', label=f'values ($m={m_val}$)')
        res_k, std_k, res_b, std_b, x_log_min, x_log_max = loglog_fit(N_space, mean_k_max_space)
        fitspace = np.linspace(x_log_min, x_log_max, 100)
        plt.plot(np.exp(fitspace), np.exp(fitspace * res_k + res_b), linestyle = 'dashed', label = f'fit; $r = {res_k:.2f}\\pm{std_k:.4f}$ ($m={m_val}$)')
    
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title('Theoretical expected maximum degree for preferential attachement (m dependence)')
    plt.xlabel('Number of edges added in every step $m$')
    plt.ylabel('$\\langle k_{{max}}\\rangle (m)$')
    
    for N_val in N_space_og:
        print("Analysing N =", N_val)
        mean_k_max_space = []
        for cur_m_val in m_space:
            mean_k_max_space.append(expected_max_k(cur_m_val, N_val))
        plt.loglog(m_space, mean_k_max_space, 'x-', label=f'values ($N={N_val}$)')
        res_k, std_k, res_b, std_b, x_log_min, x_log_max = loglog_fit(m_space, mean_k_max_space)
        fitspace = np.linspace(x_log_min, x_log_max, 100)
        plt.plot(np.exp(fitspace), np.exp(fitspace * res_k + res_b), linestyle = 'dashed', label = f'fit; $r = {res_k:.2f}\\pm{std_k:.4f}$ ($N={N_val}$)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def task1_4(new_dataset_name, N_space = [5e4, 1e5, 2e5], N_m = 1, m = 3):
    
    k_max_avg_array, k_max_std_array,res_array = N_scaling_degree_distribution(new_dataset_name, N_space, N_m=N_m, m=m, PS = 'PA', bin_scale = 1.3)
    task1_4_analysis(N_m, N_space, m, k_max_avg_array, k_max_std_array,res_array)

def task1_4_load(dataset_name):
    
    PS, N_m, bin_scale, m_space, N_max_space, k_max_avg_array, k_max_std_array, res_array = load_N_scaling_degree_distribution(dataset_name, keep_descriptors = True)
    task1_4_analysis(N_m, N_max_space, m_space, k_max_avg_array, k_max_std_array, res_array)

def task1_4_analysis(N_m, N_space, m, k_max_avg_array, k_max_std_array,res_array):
    
    m = m[0]
    """for N_i in range(len(N_space)):
        x, y = logbin(res_array[N_i])
        fit_x_space = np.linspace(min(x), max(x))

        plt.loglog(x, y, label=f'k dist. ($N_{{max}} = 10^{{{np.round(np.log(N_space[N_i])/np.log(10))}}}$)')
        #plt.loglog(np.exp(fit_log_x_space), np.exp(-fit_log_x_space * 3.0 + res_b_three), linestyle='dotted', label=f'$k=-3$')
        plt.loglog(fit_x_space, theoretical_gamma_p_infty(fit_x_space, m_val), linestyle='dotted', label=f'prediction ($N_{{max}} = 10^{{{np.round(np.log(N_space[N_i])/np.log(10))}}}$)', color=plt.gca().lines[-1].get_color())
    plt.legend()
    plt.show()"""
    plt.title(f"$k_{{max}}$ as a function of the number of iterations (m = {m})")
    plt.xlabel("$N_{max}$")
    plt.ylabel("$k_{max}$")
    if sum(k_max_std_array) > 0:
        plt.errorbar(N_space, k_max_avg_array, yerr = k_max_std_array / np.sqrt(N_m), label='values')
    else:
        plt.plot(N_space, k_max_avg_array, linestyle='x-', label='values')
    
    theoretical_k_max_avg = expected_max_k(m, N_space)
    plt.plot(N_space, theoretical_k_max_avg, linestyle='dashed', label=f'prediction')
    
    plt.legend()
    plt.show()
    """plt.title("Probability distribution of node degree")
    plt.gca().set_xscale("log", base=10)
    plt.gca().set_yscale("log", base=10)
    plt.xlabel("$k$")
    plt.ylabel("$p_{\\infty}(k)$")
    for N_i in range(len(N_space)):
        x, y_PDF, y_PDF_err, y_counts, y_counts_err, binedges = res_array[N_i]
        x_nonzero = x[y_PDF!=0]
        y_nonzero = y_PDF[y_PDF!=0]
        
        yerr_nonzero = y_PDF_err[y_PDF!=0]
        
        xerr_left = x - binedges[:-1]
        xerr_right = binedges[1:] - x
        
        xerr_left_nonzero = xerr_left[y_PDF!=0]
        xerr_right_nonzero = xerr_right[y_PDF!=0]
        
        if sum(yerr_nonzero) > 0:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], yerr = yerr_nonzero, fmt='x', capsize=10, label=f'data ($N_{{max}} = {N_space[N_i]}$)')
        else:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], fmt='x', capsize=10, label=f'data ($N_{{N_max}} = {N_space[N_i]}$)')
        #plt.loglog(x, theoretical_binned_distribution_list[m_i], linestyle='dotted', label=f'prediction ($m = {m_space[m_i]}$)')
    
    plt.legend()
    plt.show()"""

#task1_3_load('1_3_5_big')
#task1_4_expected_k_max()
#task1_4('test', [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5], N_m = 5)
task1_4_load('first')

"""
test = BA_network(m = 3, probability_strategy = 'PA')
test.initial_graph(strategy = 'r', N = 10, param = 3)

test.simulate(1e4)

x, y = logbin(test.degree_distribution(uniform_bins = False))

def theoretical_p_infty(k, m):
    return((2.0 * m * (m + 1.0)) / (k * (k + 1.0) * (k + 2.0)))


# the -0.5 * (m + 0.5) scaling
res_b_three, std_b_three, x_min, x_max = get_b_from_line(x, y, -3.0)
fit_log_x_space = np.linspace(x_min, x_max, 100)

fit_x_space = np.linspace(min(x), max(x))

plt.loglog(x, y)
plt.loglog(np.exp(fit_log_x_space), np.exp(-fit_log_x_space * 3.0 + res_b_three), linestyle='dotted', label=f'$k=-3$')
plt.loglog(fit_x_space, theoretical_gamma_p_infty(fit_x_space, test.m), linestyle='dotted', label=f'gamma func')

plt.legend()
plt.show()"""


"""cur_hist, cur_bin_edges = test.degree_distribution()
cur_bin_centres = (cur_bin_edges[1:]+cur_bin_edges[:-1])/2.0
cur_bin_widths = cur_bin_edges[1:]-cur_bin_edges[:-1]

plt.plot(cur_bin_centres,cur_hist, '-')
plt.errorbar(cur_bin_centres, cur_hist, xerr=(cur_bin_widths / 2.0) , fmt='x', label=f'lol')
plt.show()"""

#print(test)



"""
lol = []
for i in range(17000):
    lol.append(sum(weighted_sample([1.0, 1.0, 10.0], 2)))

plt.hist(lol)
plt.show()"""
