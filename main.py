from BA_network_class import *


# -----------------------------------------------------------
# -------------------- Fitting functions --------------------
# -----------------------------------------------------------    

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

# -----------------------------------------------------------
# ---------------- Scaling test functions -------------------
# -----------------------------------------------------------

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

# special scaling test for degree distribution that handles the statistical data churning on the ugly logbinned arrays
def m_scaling_degree_distribution(dataset_name = 'dataset', m_space = [3], r_space = -1, N_m=1, PS = 'PA', N_max = 1e4, bin_scale = 1.3):
    # m_space = array of m values to test on
    # r_space is also interpreted as the minimum cutoff in logbinning for all models
    # N_m = number of repetitions per m value datapoint
    # PS is the strategy
    # N_max = final number of nodes - THIS CAN BE A LIST
    if type(N_max) != list:
        N_max = [N_max] * len(m_space)
    if r_space == -1:
        r_space = m_space
    res_array = []
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
            cur_BA_network = BA_network(m = m_val, probability_strategy = PS, r = r_space[m_i])
            
            degree_array = cur_BA_network.get_degree_distribution(N_max = N_max[m_i])
            x, y_PDF, y_counts, binedges = logbin(degree_array, scale = bin_scale, x_min = r_space[m_i])
            
            x_matrix.append(x)
            y_PDF_matrix.append(y_PDF)
            y_counts_matrix.append(y_counts)
            binedges_matrix.append(binedges)
            
            if max(x) > x_max:
                x_max = max(x)
                widest_sample_index = i
        
        if N_m == 1:
            res_array.append([x_matrix[0].copy(),y_PDF_matrix[0], np.zeros(len(y_PDF_matrix[0])),y_counts_matrix[0], np.zeros(len(y_counts_matrix[0])),binedges_matrix[0]])
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
            res_array.append([x,y_PDF_avg,y_PDF_std,y_counts_avg,y_counts_std,binedges])
        """if N_m > 1 :
            res_array.append(np.average(res_matrix, axis=0))
            res_std_array.append(np.std(res_matrix, axis=0))
        else:
            res_array.append(res_matrix[0])"""
        #print(f"  Result = {res_array[-1]}")
    
    # We now save the res_array into a txt file
    save_m_scaling_degree_distribution(dataset_name, PS, N_m, bin_scale, m_space, r_space, N_max, res_array)
    return(res_array)
    
def OLD_N_scaling_degree_distribution(dataset_name = 'dataset', N_space = [1e4], N_m=1, m = 3, r = 3, PS = 'PA', bin_scale = 1.3):
    # N_space = array of N_max values to test on
    # N_m = number of repetitions per m value datapoint
    if type(m) != list:
        m = [m] * len(N_space)
    res_array = []
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
            cur_BA_network = BA_network(m = m[N_i], probability_strategy = PS, r = r)
            
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
            res_array.append([x_matrix[0].copy(),y_PDF_matrix[0], np.zeros(len(y_PDF_matrix[0])),y_counts_matrix[0], np.zeros(len(y_counts_matrix[0])),binedges_matrix[0]])
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
            res_array.append([x,y_PDF_avg,y_PDF_std,y_counts_avg,y_counts_std,binedges])
            k_max_avg_array.append(np.average(cur_k_max_array))
            k_max_std_array.append(np.std(cur_k_max_array))
    
    # We now save the res_array into a txt file
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
    N_scaling_degree_distribution_output_file = open("data/N_k_scaling_" + PS + "_" + dataset_name + ".txt", mode="w")
    #filename_list_stringed = 'L'.join([str(elem) for elem in my_filename_list])
    #meta_config_file.write(filename_list_stringed)
    N_scaling_degree_distribution_output_file.write(f'{PS}, {N_m}, {bin_scale}\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in m]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in N_space]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in k_max_avg_array]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in k_max_std_array]) + '\n')
    for N_i in range(len(N_space)):
        for res_i in range(len(res_array[N_i])):
            N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in res_array[N_i][res_i]]) + '\n')
    N_scaling_degree_distribution_output_file.close()
    
    return(k_max_avg_array, k_max_std_array,res_array)
    
def N_scaling_degree_distribution(dataset_name = 'dataset', N_space = [1e4], N_m=1, m = 3, r = -1, PS = 'PA', bin_scale = 1.3):
    # N_space = array of N_max values to test on
    # N_m = number of repetitions per m value datapoint
    if r == -1:
        r = m
    
    res_array = []
    k_max_avg_array = []
    k_max_std_array = []
    print(f"--------- Commencing testing on {len(N_space)} instances (dataset '{dataset_name}') --------")
        
    x_matrix = []
    y_PDF_matrix = []
    y_counts_matrix = []
    binedges_matrix = []
    widest_sample_index = []
    x_max = []
    cur_k_max_array = []
    
    for N_i in range(len(N_space)):
        x_matrix.append([])
        y_PDF_matrix.append([])
        y_counts_matrix.append([])
        binedges_matrix.append([])
        widest_sample_index.append(-1)
        x_max.append(0)
        cur_k_max_array.append([])
    
    for i in range(N_m):
        print(f"  current index = {i+1}/{N_m}")
        cur_BA_network = BA_network(m = m, probability_strategy = PS, r = r)
        
        degree_array_list = cur_BA_network.get_degree_distribution(N_max = N_space)
        
        for N_i in range(len(N_space)):
            degree_array = degree_array_list[N_i]
        
            cur_k_max_array[N_i].append(max(degree_array))
            x, y_PDF, y_counts, binedges = logbin(degree_array, scale = bin_scale, x_min = r)
            
            x_matrix[N_i].append(x)
            y_PDF_matrix[N_i].append(y_PDF)
            y_counts_matrix[N_i].append(y_counts)
            binedges_matrix[N_i].append(binedges)
            
            if max(x) > x_max[N_i]:
                x_max[N_i] = max(x)
                widest_sample_index[N_i] = i
            
    for N_i in range(len(N_space)):
        if N_m == 1:
            res_array.append([x_matrix[N_i][0].copy(),y_PDF_matrix[N_i][0], np.zeros(len(y_PDF_matrix[N_i][0])),y_counts_matrix[N_i][0], np.zeros(len(y_counts_matrix[N_i][0])),binedges_matrix[N_i][0]])
            k_max_avg_array.append(cur_k_max_array[N_i][0])
            k_max_std_array.append(0.0)
        else:
            # calculate the ugly statisics
            
            # first, locate the x-array and binedges-array that spans the entire dataset
            x = x_matrix[N_i][widest_sample_index[N_i]]
            binedges = binedges_matrix[N_i][widest_sample_index[N_i]]
            print("Length of sample-spanning x-array =", len(x))
            # now append empty bins to all lacking y-arrays - assume the non-spanning binedges are always left-aligned
            for i in range(N_m):
                length_difference = len(x) - len(x_matrix[N_i][i])
                if length_difference > 0:
                    y_PDF_matrix[N_i][i] = np.concatenate((y_PDF_matrix[N_i][i], np.zeros(length_difference)))
                    y_counts_matrix[N_i][i] = np.concatenate((y_counts_matrix[N_i][i], np.zeros(length_difference)))
            # now calculate the statistics
            y_PDF_avg = np.average(y_PDF_matrix[N_i], axis=0)
            y_PDF_std = np.std(y_PDF_matrix[N_i], axis=0)
            y_counts_avg = np.average(y_counts_matrix[N_i], axis=0)
            y_counts_std = np.std(y_counts_matrix[N_i], axis=0)
            res_array.append([x,y_PDF_avg,y_PDF_std,y_counts_avg,y_counts_std,binedges])
            k_max_avg_array.append(np.average(cur_k_max_array[N_i]))
            k_max_std_array.append(np.std(cur_k_max_array[N_i]))
    
    # We now save the res_array into a txt file
    save_N_scaling_degree_distribution(dataset_name, PS, N_m, m, r, bin_scale, N_space, k_max_avg_array, k_max_std_array, res_array)
    return(k_max_avg_array, k_max_std_array,res_array)

# -----------------------------------------------------------
# ------------------ Filestream functions -------------------
# -----------------------------------------------------------

def encode_PS(PS):
    if type(PS) == list:
        return("_".join(PS))
    return(PS)
def decode_PS(PS):
    if "_" in PS:
        return PS.split("_")
    return(PS)

def save_m_scaling_degree_distribution(dataset_name, PS, N_m, bin_scale, m_space, r_space, N_max_space, res_array):
    """
    The file has the form:
        1 [PS, N_m, bin_scale]
        2 [m1, m2, m3...]
        3 [r1, r2, r3...]
        4 [N_max1, N_max2, N_max3...]
        Then for each m_val, there are 6 rows:
            1 [x]
            2 [y_PDF]
            3 [y_PDF_std]
            4 [y_counts]
            5 [y_counts_std]
            5 [binedges]
    """
    PS_encoded = encode_PS(PS)
    output_delim = ', '
    m_scaling_degree_distribution_output_file = open("data/m_k_scaling_" + PS_encoded + "_" + dataset_name + ".txt", mode="w")
    m_scaling_degree_distribution_output_file.write(f'{PS_encoded}, {N_m}, {bin_scale}\n')
    m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in m_space]) + '\n')
    m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in r_space]) + '\n')
    m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in N_max_space]) + '\n')
    for m_i in range(len(m_space)):
        for res_i in range(len(res_array[m_i])):
            m_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in res_array[m_i][res_i]]) + '\n')
    m_scaling_degree_distribution_output_file.close()


def save_N_scaling_degree_distribution(dataset_name, PS, N_m, m, r, bin_scale, N_max_space, k_max_avg_array, k_max_std_array, res_array):
    """
    The file has the form:
        1 [PS, N_m, m, r, bin_scale]
        2 [N_max1, N_max2, N_max3...]
        3 [k_max1, k_max2, k_max3...]
        4 [k_max_std1, k_max_std2, k_max_std3...]
        Then for each m_val, there are 6 rows:
            1 [x]
            2 [y_PDF]
            3 [y_PDF_std]
            4 [y_counts]
            5 [y_counts_std]
            5 [binedges]
    """
    PS_encoded = encode_PS(PS)
    output_delim = ', '
    N_scaling_degree_distribution_output_file = open("data/N_k_scaling_" + PS_encoded + "_" + dataset_name + ".txt", mode="w")
    N_scaling_degree_distribution_output_file.write(f'{PS_encoded}, {N_m}, {m}, {r}, {bin_scale}\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in N_max_space]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in k_max_avg_array]) + '\n')
    N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in k_max_std_array]) + '\n')
    for N_i in range(len(N_max_space)):
        for res_i in range(len(res_array[N_i])):
            N_scaling_degree_distribution_output_file.write(output_delim.join([str(elem) for elem in res_array[N_i][res_i]]) + '\n')
    N_scaling_degree_distribution_output_file.close()

def load_m_scaling_degree_distribution(dataset_name, keep_descriptors = False):
    
    # Input file configuration parameters
    head_line_number = 4
    line_number_per_m_val = 6
    data_line_types = [float, float, float, float, float, int]
    
    m_scaling_degree_distribution_input_file = open("data/m_k_scaling_" + dataset_name + ".txt", mode="r")
    input_lines = m_scaling_degree_distribution_input_file.readlines()
    m_scaling_degree_distribution_input_file.close()
    
    head_lines = [item.split(", ") for item in input_lines[:head_line_number]]
    #data_lines = [[float(elem) for elem in item.split(", ")] for item in input_lines[head_line_number:]]
    data_lines = [item.split(", ") for item in input_lines[head_line_number:]]
    
    PS = decode_PS(head_lines[0][0])
    N_m = int(head_lines[0][1])
    bin_scale = float(head_lines[0][2])
    
    m_space = [int(elem) for elem in head_lines[1]]
    r_space = [int(elem) for elem in head_lines[2]]
    N_max_space = [float(elem) for elem in head_lines[3]]
    
    
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
        return(PS, N_m, bin_scale, m_space, r_space, N_max_space, res_array)
    return(res_array)

def load_N_scaling_degree_distribution(dataset_name, keep_descriptors = False):
    
    # Input file configuration parameters
    head_line_number = 4
    line_number_per_m_val = 6
    data_line_types = [float, float, float, float, float, int]
    
    N_scaling_degree_distribution_input_file = open("data/N_k_scaling_" + dataset_name + ".txt", mode="r")
    input_lines = N_scaling_degree_distribution_input_file.readlines()
    N_scaling_degree_distribution_input_file.close()
    
    head_lines = [item.split(", ") for item in input_lines[:head_line_number]]
    #data_lines = [[float(elem) for elem in item.split(", ")] for item in input_lines[head_line_number:]]
    data_lines = [item.split(", ") for item in input_lines[head_line_number:]]
    
    PS = decode_PS(head_lines[0][0])
    N_m = int(head_lines[0][1])
    m = int(head_lines[0][2])
    r = int(head_lines[0][3])
    bin_scale = float(head_lines[0][4])
    
    N_max_space = [float(elem) for elem in head_lines[1]]
    
    k_max_avg_array = np.array([float(elem) for elem in head_lines[2]])
    k_max_std_array = np.array([float(elem) for elem in head_lines[3]])
    
    
    res_array = []
    try:
        for N_i in range(len(N_max_space)):
            res_array.append([])
            for dataline_i in range(line_number_per_m_val):
                res_array[N_i].append(np.array([data_line_types[dataline_i](item) for item in data_lines[N_i * line_number_per_m_val + dataline_i]]))
    except IndexError:
        print("ERROR: Input file broken")
        return(-1)
        
    
    print(f"--------------- Loaded dataset '{dataset_name}' --------------")
    print(f"Probability strategy: {PS}; Number of repetitions per datapoint = {N_m}; m = {m}; Bin scale = {bin_scale}")
    print(f"N_max_space = {N_max_space}")
    if keep_descriptors:
        return(PS, N_m, bin_scale, m, r, N_max_space, k_max_avg_array, k_max_std_array, res_array)
    return(k_max_avg_array, k_max_std_array, res_array)

def combine_datasets(ds1, ds2, which_scaling, ds_out = -1, PS = 'PA'):
    # If ds1 and ds2 are compatible datasets, this creates a combined dataset and saves it into ds_out
    # which_scaling determines whether this is m_k_scaling ('m') or N_k_scaling ('N')
    
    def weighted_mean(a_avg1, a_err1, a_avg2, a_err2, N_m1, N_m2):
        a_avg3 = (N_m1 * a_avg1 + N_m2 * a_avg2) / (N_m1 + N_m2)
        a_err3 = np.sqrt(N_m1 * N_m1 * a_err1 * a_err1 + N_m2 * N_m2 * a_err2 * a_err2) / (N_m1 + N_m2)
        return(a_avg3, a_err3)
    
    def combine_res_arrays(res_array1, res_array2, N_m1, N_m2, N):
        # oomph
        # 'N' is the number of distinct measurements - these must match
        # it's always [x, y_PDF, y_PDF_std, y_counts, y_counts_std, binedges]
        res_array3 = []
        for N_i in range(N):
            x1, y_PDF1, y_PDF_err1, y_counts1, y_counts_err1, binedges1 = res_array1[N_i]
            x2, y_PDF2, y_PDF_err2, y_counts2, y_counts_err2, binedges2 = res_array2[N_i]
            if len(x1) < len(x2):
                len_diff = len(x2) - len(x1)
                x3        = x2.copy()
                binedges3 = binedges2.copy()
                y_PDF1        = np.concatenate((y_PDF1       , np.zeros(len_diff)))
                y_PDF_err1    = np.concatenate((y_PDF_err1   , np.zeros(len_diff)))
                y_counts1     = np.concatenate((y_counts1    , np.zeros(len_diff)))
                y_counts_err1 = np.concatenate((y_counts_err1, np.zeros(len_diff)))
            elif len(x2) < len(x1):
                len_diff = len(x1) - len(x2)
                x3        = x1.copy()
                binedges3 = binedges1.copy()
                y_PDF2        = np.concatenate((y_PDF2       , np.zeros(len_diff)))
                y_PDF_err2    = np.concatenate((y_PDF_err2   , np.zeros(len_diff)))
                y_counts2     = np.concatenate((y_counts2    , np.zeros(len_diff)))
                y_counts_err2 = np.concatenate((y_counts_err2, np.zeros(len_diff)))
            else:
                x3        = x1.copy()
                binedges3 = binedges1.copy()
            # since the binedges must be the same, we can literally perform a weighted mean everywhere
            y_PDF3   , y_PDF_err3    = weighted_mean(y_PDF1   , y_PDF_err1   , y_PDF2   , y_PDF_err2   , N_m1, N_m2)
            y_counts3, y_counts_err3 = weighted_mean(y_counts1, y_counts_err1, y_counts2, y_counts_err2, N_m1, N_m2)
            res_array3.append([x3, y_PDF3, y_PDF_err3, y_counts3, y_counts_err3, binedges3])
        return(res_array3)
    
    if ds_out == -1:
        ds_out = "COMBINED_" + ds1 + "_" + ds2
    
    if which_scaling.lower() == 'm':
        PS1, N_m1, bin_scale1, m_space1, r_space1, N_max_space1, res_array1 = load_m_scaling_degree_distribution(PS + "_" + ds1, keep_descriptors = True)
        PS2, N_m2, bin_scale2, m_space2, r_space2, N_max_space2, res_array2 = load_m_scaling_degree_distribution(PS + "_" + ds2, keep_descriptors = True)
        
        if bin_scale1 != bin_scale2 or m_space1 != m_space2 or r_space1 != r_space2 or N_max_space1 != N_max_space2:
            print(f"ERROR: Datasets '{ds1}' and '{ds2}' found incompatible!")
            return(-1)
        N_m3 = N_m1 + N_m2
        bin_scale3 = bin_scale1
        m_space3 = m_space1
        r_space3 = r_space1
        N_max_space3 = N_max_space1
        res_array3 = combine_res_arrays(res_array1, res_array2, N_m1, N_m2, len(N_max_space3))
        
        save_m_scaling_degree_distribution(ds_out, PS, N_m3, bin_scale3, m_space3, r_space3, N_max_space3, res_array3)
        return(PS, N_m3, bin_scale3, m_space3, r_space3, N_max_space3, res_array3)
        
        
    if which_scaling.lower() == 'n':
        PS1, N_m1, bin_scale1, m1, r1, N_max_space1, k_max_avg_array1, k_max_std_array1, res_array1 = load_N_scaling_degree_distribution(encode_PS(PS) + "_" + ds1, keep_descriptors = True)
        PS2, N_m2, bin_scale2, m2, r2, N_max_space2, k_max_avg_array2, k_max_std_array2, res_array2 = load_N_scaling_degree_distribution(encode_PS(PS) + "_" + ds2, keep_descriptors = True)
        
        if bin_scale1 != bin_scale2 or m1 != m2 or r1 != r2 or N_max_space1 != N_max_space2:
            print(f"ERROR: Datasets '{ds1}' and '{ds2}' found incompatible!")
            return(-1)
        
        N_m3 = N_m1 + N_m2
        m3 = m1
        r3 = r1
        bin_scale3 = bin_scale1
        N_max_space3 = N_max_space1
        k_max_avg_array3, k_max_std_array3 = weighted_mean(k_max_avg_array1, k_max_std_array1, k_max_avg_array2, k_max_std_array2, N_m1, N_m2)
        res_array3 = combine_res_arrays(res_array1, res_array2, N_m1, N_m2, len(N_max_space3))
        
        save_N_scaling_degree_distribution(ds_out, PS, N_m3, m3, r3, bin_scale3, N_max_space3, k_max_avg_array3, k_max_std_array3, res_array3)
        return(PS, N_m3, bin_scale3, m3, r3, N_max_space3, k_max_avg_array3, k_max_std_array3, res_array3)
        

# -----------------------------------------------------------
# ----------- Theoretical prediction functions --------------
# -----------------------------------------------------------

def p_infty(k, m, r, PS = 'PA'):
    if PS == 'PA':
        return((2.0 * m * (m + 1.0)) / (k * (k + 1.0) * (k + 2.0)))
    if PS == 'RA':
        return(np.power(m / (m + 1.0), k - m) / (m + 1.0))
    if PS == ['RA', 'PA']:
        return(np.power(k + r * m / (m - r), - (2.0 * m - r) / (m - r)) / sp.special.zeta((2.0 * m - r) / (m - r), r * (2.0 * m - r) / (m - r)))

def primitive_function_p_infty(k, m, r, PS = 'PA'):
    # This is the promitive function of theoretical_gamma_p_infty(k, m)
    if PS == 'PA':
        return(m * (m + 1.0) * (np.log(k) - 2.0 * np.log(k + 1.0) + np.log(k + 2.0)))
    if PS == 'RA':
        return(np.power(m / (m + 1.0), k - m) / ((m + 1.0) * np.log(m / (m + 1.0))))
    if PS == ['RA', 'PA']:
        #return(- ((m - r) / m) * np.power(k + r * m / (m - r), -m / (m - r)) / sp.special.zeta((2.0 * m - r) / (m - r), r * (2.0 * m - r) / (m - r)))
        return(- ((m - r) / m) * np.power(k + r * m / (m - r), -m / (m - r)) * (m / (m - r)) * np.power(r * (2.0 * m - r) / (m - r), m / (m - r)))

"""def expected_frequency_p_infty(k, m):
    # this is a discrete version of theoretical_gamma_p_infty
    if k == m:
        return(2.0 / (m + 2.0))
    else:
        return((2.0 * m * (m + 1.0)) / (k * (k + 1.0) * (k + 2.0)))"""

def expected_bin_frequency_p_infty(m, r, k_min, k_max = -1, PS = 'PA'):
    # counts the theoretical frequency for k_min <= k < k_max
    # if k_max == -1, it is interpreted as infinity
    if PS == 'PA':
        if k_max == -1:
            return(m * (m + 1.0) / (k_min * (k_min + 1.0)))
        return(m * (m + 1.0) * (k_max * (k_max + 1.0) - k_min * (k_min + 1.0)) / (k_max * (k_max + 1.0) * k_min * (k_min + 1.0)))
    if PS == 'RA':
        power_base = m / (m + 1.0)
        if k_max == -1:
            return(np.power(power_base, k_min - m))
        return(np.power(power_base, k_min - m) - np.power(power_base, k_max - m))
    if PS == ['RA', 'PA']:
        
        #if k_max == -1:
        #    return( sp.special.zeta((2.0 * m - r) / (m - r), r * m / (m - r) + k_min) / sp.special.zeta((2.0 * m - r) / (m - r), r * (2.0 * m - r) / (m - r)) )
        #return( (sp.special.zeta((2.0 * m - r) / (m - r), r * m / (m - r) + k_min) - sp.special.zeta((2.0 * m - r) / (m - r), r * m / (m - r) + k_max)) / sp.special.zeta((2.0 * m - r) / (m - r), r * (2.0 * m - r) / (m - r)) )
        # the discrete solution:
        return(discrete_expected_bin_frequency_EVM(m, r, k_min, k_max))
        """A = m / (2.0 * m * r + m - r * r) * sp.special.gamma((2.0 * m * r + 3.0 * m - 2.0 * r - r * r)/(m - r)) / sp.special.gamma((2.0 * m * r + 2.0 * m - 2.0 * r - r * r)/(m - r))
        if k_min == 1 and k_max == 2:
            return(m / (2 * m * r + m - r * r))
        elif k_min == 1:
            return(m / (2 * m * r + m - r * r) + expected_bin_frequency_p_infty(m, r, k_min + 1, k_max, PS))
        elif k_max == -1:
            return()"""

def discrete_expected_bin_frequency_EVM(m, r, k_min, k_max):
    # this is the sus impostor function
    if type(k_min) == list or type(k_min) == np.ndarray:
        # iterable
        res_array = []
        if k_max == -1:
            k_max_new = [-1] * len(k_min)
        else:
            k_max_new = k_max
        for i in range(len(k_min)):
            res_array.append(discrete_expected_bin_frequency_EVM(m, r, k_min[i], k_max_new[i]))
        return(np.array(res_array))
    if k_min == r:
        return(m / (2.0 * m * r + m - r * r) + discrete_expected_bin_frequency_EVM(m, r, k_min+1, k_max))
    z1 = m * r / (m - r)
    z2 = z1 + (2.0 * m - r) / (m - r)
    # prod 1
    prod1 = 1.0
    if k_min > r + 1:
        for i in range(r+1, k_min):
            prod1 *= (z1 + i) / (z2 + i)
    # prod 2
    if k_max == -1:
        prod2 = 0.0
    else:
        prod2 = 1.0
        for i in range(r+1, k_max):
            prod2 *= (z1 + i) / (z2 + i)
    factor = m * r / ((r + 1.0) * (2.0 * m * r + m - r * r))
    return(factor * (prod1 * (1.0 + r + k_min - r * k_min / m) + prod2 * (r * (k_max-1.0) / m - k_max + r / m - r - 1.0)))
    
        
def theoretical_binned_p_infty(bin_edges, m, r, PS = 'PA'):
    # This calculates the expected normalized AND unnormalized binned distribution of the gamma func. p_infty function
    # y[i] = (bin_edges[i+1] - bin_edges[i])^-1 * \int_{bin_edges[i]}^{bin_edges[i+1]} f(k) dk
    
    #if PS == 'PA':
    y_PDF = (primitive_function_p_infty(bin_edges[1:], m, r, PS) - primitive_function_p_infty(bin_edges[:-1], m, r, PS)) / (bin_edges[1:] - bin_edges[:-1])
    y_counts = np.zeros(len(bin_edges)-1)
    for i in range(len(bin_edges)-1):
        y_counts[i] = expected_bin_frequency_p_infty(m, r, bin_edges[i], bin_edges[i+1], PS)
    #elif PS == 'RA':
        
    """for i in range(len(bin_edges)-2):
        y_counts[i] = expected_bin_frequency_p_infty(m, r, bin_edges[i], bin_edges[i+1])
    y_counts[-1] = expected_bin_frequency_p_infty(m, r, bin_edges[-2], -1)"""
    return(y_PDF, y_counts)
    # note that neither of these is normalized - the PDF is non-negligible even for bins outside of the bin_edges scope

def independent_sampling_expected_max_k(m, N, k_max = 1000000):
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

def expected_max_k(m, r, N, PS = 'PA'):
    if PS == 'PA':
        return((-1.0 + np.sqrt(1 + 4.0 * N * m * (m + 1.0)))/2.0)
    if PS == 'RA':
        return(m + np.log(N) / np.log((m + 1.0) / m))
    if PS == ['RA', 'PA']:
        # this is a bit sussy
        return(discrete_expected_max_k_EVM(m, r, N))

def discrete_expected_max_k_EVM(m, r, N):
    if type(N) == list or type(N) == np.ndarray:
        # iterable
        res_array = []
        for i in range(len(N)):
            res_array.append(discrete_expected_max_k_EVM(m, r, N[i]))
        return(np.array(res_array))
    # we basically do a search in a monotone function for a specific value
    # TODO the following algorithm is terribly inefficient
    cur_k_max = r+1
    while(True):
        cur_sum_prob = discrete_expected_bin_frequency_EVM(m, r, cur_k_max, -1)
        if cur_sum_prob < 1.0 / N:
            return(cur_k_max)
        cur_k_max += 1

def approximate_expected_max_k_EVM(m, r, N):
    return((np.power(N, (m-r)/m) * r * (2 * m - r) - r * m) / (m - r))

# -----------------------------------------------------------
#----------------- Data analysis functions ------------------
# -----------------------------------------------------------

def weighted_chi_sq(y_measurement, y_prediction, y_std):  
      chi_sq = np.sum( ((y_measurement-y_prediction)/y_std)**2/y_prediction )  
      return(chi_sq)

def sanitize_fat_tail(bincounts, binedges, theoretical_binned_frequency, m_val, r_val, min_bin_count = 5, PS = 'PA'):
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
        return(bincounts, binedges, np.concatenate((theoretical_binned_frequency[:-1], [theoretical_binned_frequency[-1] + expected_bin_frequency_p_infty(m_val, r_val, max(binedges), PS=PS)])))
    new_bincounts = np.concatenate((bincounts[:cutoff_index], [cum_bin_count]))
    new_binedges = binedges.copy()[:cutoff_index+1] #this implies an 'infinity' as a final entry, which we don't include, ofc
    # NOTE the goodness-of-fit doesn't actually take into account the x-position of the stuff. That's okay i guess. Well it does in the sense the theoretical prediction is an x sum
    new_theoretical_binned_frequency = np.concatenate((theoretical_binned_frequency[:cutoff_index], [expected_bin_frequency_p_infty(m_val, r_val,new_binedges[-1], PS = PS)]))
    print(f'  LOG: {len(bincounts) - cutoff_index} bins in the fat tail combined; new minimum bincount is {int(min(new_bincounts))}')
    return(new_bincounts, new_binedges, new_theoretical_binned_frequency)

def k_degree_distribution_analysis(PS, N_m, m_space, r_space, N_max, res_array):
    
    # res_array obtained either by m_scaling_degree_distribution or by load_m_scaling_degree_distribution
    
    print("---------------------------------------------------- ")
    print("--- Analysis of a degree distribution commencing ---")
    print("----------------------------------------------------")
    
    print(f"Probability strategy = {PS}")
    
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
        theoretical_binned_distribution, theoretical_binned_frequency = theoretical_binned_p_infty(binedges, m_space[m_i], r_space[m_i], PS = PS)
        print("  Sum of theoretical binned frequency counts =", sum(theoretical_binned_frequency))
        
        # theoretical distribution is normalized on m <= k <= infty, but in the measurement we have the N_max limitation, which
        # omits these high values. Hence we will extend the last bin in our counts (theoretical and measured), which has borders at
        # binedges[-2] and infinity. (basically setting binedges[-1] to infinity)
        
        #theoretical_binned_frequency_with_infinity = np.concatenate((theoretical_binned_frequency[:-1], [theoretical_binned_frequency[-1] + expected_bin_frequency_p_infty(m_space[m_i], max(binedges))]))
        #y_counts_with_infinity = y_counts.copy()#np.concatenate((y_counts, [0.0]))
        #NOTE we now do the "cumulative infinity binning"
        y_counts_with_infinity, binedges_with_infinity, theoretical_binned_frequency_with_infinity = sanitize_fat_tail(y_counts, binedges, theoretical_binned_frequency, m_space[m_i], r_space[m_i], PS=PS)
        print("  sum of theoretical binned frequency INCLUDING INFINITY =", sum(theoretical_binned_frequency_with_infinity))
        
        
        #loglog power fitting
        x_nonzero = x[y_PDF!=0]
        y_nonzero = y_PDF[y_PDF!=0]
        fit_res_k, fit_std_k, fit_res_b, fit_std_b, fit_x_log_min, fit_x_log_max= loglog_fit(x_nonzero, y_nonzero)
        print(f"Measured power law: p(k) goes like k^(-{-fit_res_k}+-{fit_std_k})")
        print(f"(2m-r)/(m-r)={(2.0*m_space[m_i]-r_space[m_i])/(m_space[m_i]-r_space[m_i])}")
        
        
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
        theoretical_binned_distribution_list.append(theoretical_binned_distribution)
        theoretical_binned_frequency_list.append(theoretical_binned_frequency)
    
    plt.subplot(2, 1, 1)
    plt.title("Probability distribution of node degree")
    
    plt.xlabel("$k$")
    plt.ylabel("$p_{\\infty}(k)$")
    for m_i in range(len(m_space)):
        x, y_PDF, y_PDF_err, y_counts, y_counts_err, binedges = res_array[m_i]
        x_nonzero = x[y_PDF!=0]
        y_nonzero = y_PDF[y_PDF!=0]
        
        yerr_nonzero = y_PDF_err[y_PDF!=0]
        
        xerr_left = x - binedges[:-1]
        xerr_right = binedges[1:] - x
        
        xerr_left_nonzero = xerr_left[y_PDF!=0]
        xerr_right_nonzero = xerr_right[y_PDF!=0]
        
        if sum(yerr_nonzero) > 0:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], yerr = yerr_nonzero, fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        else:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        plt.loglog(x, theoretical_binned_distribution_list[m_i], linestyle='dotted', label=f'prediction ($m = {m_space[m_i]}$)')
    
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Bin count of node degree")
    
    plt.xlabel("k")
    plt.ylabel("$N_k$(bin)")
    
    for m_i in range(len(m_space)):
        x, y_PDF, y_PDF_err, y_counts, y_counts_err, binedges = res_array[m_i]
        x_nonzero = x[y_PDF!=0]
        y_nonzero = y_counts[y_PDF!=0]
        
        yerr_nonzero = y_counts_err[y_PDF!=0]
        
        xerr_left = x - binedges[:-1]
        xerr_right = binedges[1:] - x
        
        xerr_left_nonzero = xerr_left[y_PDF!=0]
        xerr_right_nonzero = xerr_right[y_PDF!=0]
        
        if sum(yerr_nonzero) > 0:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], yerr = yerr_nonzero, fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        else:
            plt.errorbar(x_nonzero, y_nonzero, xerr=[xerr_left_nonzero, xerr_right_nonzero], fmt='x', capsize=10, label=f'data ($m = {m_space[m_i]}$)')
        plt.loglog(x, theoretical_binned_frequency_list[m_i] * N_max[m_i], linestyle='dotted', label=f'prediction ($m = {m_space[m_i]}$)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def k_max_analysis(PS, N_m, N_space, m, r, k_max_avg_array, k_max_std_array,res_array, plot_measured_k_max = False):
    
    if type(m) == list:
        m = m[0]
    if type(r) == list:
        r = r[0]
    """for N_i in range(len(N_space)):
        x, y = logbin(res_array[N_i])
        fit_x_space = np.linspace(min(x), max(x))

        plt.loglog(x, y, label=f'k dist. ($N_{{max}} = 10^{{{np.round(np.log(N_space[N_i])/np.log(10))}}}$)')
        #plt.loglog(np.exp(fit_log_x_space), np.exp(-fit_log_x_space * 3.0 + res_b_three), linestyle='dotted', label=f'$k=-3$')
        plt.loglog(fit_x_space, theoretical_gamma_p_infty(fit_x_space, m_val), linestyle='dotted', label=f'prediction ($N_{{max}} = 10^{{{np.round(np.log(N_space[N_i])/np.log(10))}}}$)', color=plt.gca().lines[-1].get_color())
    plt.legend()
    plt.show()"""
    theoretical_k_max_avg = expected_max_k(m, r, np.array(N_space), PS)
    if PS == ['RA', 'PA']:
        approx_theoretical_k_max_avg = approximate_expected_max_k_EVM(m, r, np.array(N_space))
    if plot_measured_k_max:
        plt.title(f"$k_{{max}}$ as a function of the number of iterations (m = {m})")
        plt.xlabel("$N_{max}$")
        plt.ylabel("$k_{max}$")
        if sum(k_max_std_array) > 0:
            plt.errorbar(N_space, k_max_avg_array, yerr = k_max_std_array / np.sqrt(N_m), label='values')
        else:
            plt.plot(N_space, k_max_avg_array, linestyle='x-', label='values')
        
        plt.plot(N_space, theoretical_k_max_avg, linestyle='dashed', label=f'prediction')
        if PS == ['RA', 'PA']:
            plt.plot(N_space, approx_theoretical_k_max_avg, linestyle='dashed', label=f'prediction (approx)')
        plt.legend()
        plt.show()
    
    
    # Now for the data collapse
    
    # k_max goes like m * sqrt(N)
    # hence we want k/k_max on x-axis. F(k/k_max) on y-axis (the cutoff function)
    
    
    plt.title("Data collapse of node degrees")
    #plt.gca().set_xscale("log", base=10)
    #plt.gca().set_yscale("log", base=10)
    plt.xlabel("$k/k_{max}$")
    plt.ylabel("$\\mathcal{{F}}\\left(k/k_{{\\mathrm{{max}}}}\\right) \\equiv p_N(k)\\cdot p_{\\infty}^{-1}(k)$")
    for N_i in range(len(N_space)):
        x, y_PDF, y_PDF_err, y_counts, y_counts_err, binedges = res_array[N_i]
        
        
        #theoretical_binned_distribution, theoretical_binned_frequency = theoretical_binned_p_infty(binedges, m, r)
        #print("  Sum of theoretical binned frequency counts =", sum(theoretical_binned_frequency))
        #y_counts_with_infinity, binedges_with_infinity, theoretical_binned_frequency_with_infinity = sanitize_fat_tail(y_counts, binedges, theoretical_binned_frequency, m)
        #print("  sum of theoretical binned frequency INCLUDING INFINITY =", sum(theoretical_binned_frequency_with_infinity))
        
        
        x_nonzero = x[y_PDF!=0]
        y_nonzero = y_PDF[y_PDF!=0]
        
        yerr_nonzero = y_PDF_err[y_PDF!=0]
        
        xerr_left = x - binedges[:-1]
        xerr_right = binedges[1:] - x
        
        xerr_left_nonzero = xerr_left[y_PDF!=0]
        xerr_right_nonzero = xerr_right[y_PDF!=0]
        
        x_factor = 1.0 / theoretical_k_max_avg[N_i] # shouldn't this be the EXPERIMENTAL k_max??
        y_factor = 1.0 / p_infty(x, m, r, PS=PS)#x_nonzero * (x_nonzero + 1.0) * (x_nonzero + 2.0) / (2.0 * m * (m + 1.0))
        
        x_collapsed = x * x_factor
        y_collapsed = y_PDF * y_factor
        
        y_collapsed_err = y_PDF_err * y_factor / np.sqrt(N_m)
        
        plt.errorbar(x_collapsed, y_collapsed, yerr = y_collapsed_err, fmt='-x', capsize=2, label=f'$N_{{\\mathrm{{max}}}} = {Decimal(N_space[N_i]):.1e}$')
        # since the y-factor is literally 1/p_infty(k), and the theoretical bin dist is p_infty(k), the prediction is just 1.0 (which is what F_N(x) tends to with N->infty)
        if N_i == len(N_space) - 1:
            plt.plot(x * x_factor, [1.0] * len(x), linestyle='dotted', label=f'$\\mathcal{{F}}(N\\to\\infty)=1$')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# ---------------------- Task functions ---------------------
# -----------------------------------------------------------

def task1_3(new_dataset_name, m_space = [1, 3, 5], N_max = [5e4, 1e5, 2e5], N_m = 1):
    
    if type(N_max) != list:
        N_max = [N_max] * len(m_space)
    res_array = m_scaling_degree_distribution(new_dataset_name, m_space, m_space, N_m=N_m, PS = 'PA', N_max = N_max, bin_scale = 1.3)
    k_degree_distribution_analysis('PA', N_m, m_space, m_space, N_max, res_array)

def task1_3_load(dataset_name):
    
    PS, N_m, bin_scale, m_space, r_space, N_max_space, res_array = load_m_scaling_degree_distribution('PA_' + dataset_name, keep_descriptors = True)
    k_degree_distribution_analysis(PS, N_m, m_space, r_space, N_max_space, res_array)

def task1_4_expected_k_max(m_space_og = [1, 3, 5], N_space_og = [1e2, 1e3, 1e4, 1e5]):
    
    # this is just the power law fitting of the theoretical k_max in task 1-4. Cannot be generalized to RA
    
    N_space = np.power(2.0, np.arange(20))#np.array([5, 1e1, 1e2, 1e3, 1e4, 1e5])
    m_space = np.arange(3, 15, 2).astype('int')
    
    plt.subplot(2, 1, 1)
    plt.title('Theoretical expected maximum degree for preferential attachement (N dependence)')
    plt.xlabel('Sample size $N$')
    plt.ylabel('$\\langle k_{{max}}\\rangle (N)$')
    
    for m_val in m_space_og:
        print("Analysing m =", m_val)
        mean_k_max_space = expected_max_k(m_val, m_val, N_space, PS = 'PA')
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
            mean_k_max_space.append(expected_max_k(cur_m_val, cur_m_val, N_val, PS = 'PA'))
        plt.loglog(m_space, mean_k_max_space, 'x-', label=f'values ($N={N_val}$)')
        res_k, std_k, res_b, std_b, x_log_min, x_log_max = loglog_fit(m_space, mean_k_max_space)
        fitspace = np.linspace(x_log_min, x_log_max, 100)
        plt.plot(np.exp(fitspace), np.exp(fitspace * res_k + res_b), linestyle = 'dashed', label = f'fit; $r = {res_k:.2f}\\pm{std_k:.4f}$ ($N={N_val}$)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def task1_4(new_dataset_name, N_space = [5e4, 1e5, 2e5], N_m = 1, m = 3):
    
    k_max_avg_array, k_max_std_array,res_array = N_scaling_degree_distribution(new_dataset_name, N_space, N_m=N_m, m=m, PS = 'PA', bin_scale = 1.3)
    k_max_analysis('PA', N_m, N_space, m, m, k_max_avg_array, k_max_std_array,res_array)

def task1_4_load(dataset_name, plot_measured_k_max = False):
    
    PS, N_m, bin_scale, m, r, N_max_space, k_max_avg_array, k_max_std_array, res_array = load_N_scaling_degree_distribution('PA_' + dataset_name, keep_descriptors = True)
    k_max_analysis(PS, N_m, N_max_space, m, r, k_max_avg_array, k_max_std_array, res_array, plot_measured_k_max = plot_measured_k_max)

def task2_1(new_dataset_name, m_space = [1, 3, 5], N_max = [5e3, 1e4, 2e4], N_m = 1):
    
    if type(N_max) != list:
        N_max = [N_max] * len(m_space)
    res_array = m_scaling_degree_distribution(new_dataset_name, m_space, N_m=N_m, PS = 'RA', N_max = N_max, bin_scale = 1.3)
    k_degree_distribution_analysis('RA', N_m, m_space, m_space, N_max, res_array)

def task2_1_load(dataset_name):
    
    PS, N_m, bin_scale, m_space, r_space, N_max_space, res_array = load_m_scaling_degree_distribution('RA_' + dataset_name, keep_descriptors = True)
    k_degree_distribution_analysis(PS, N_m, m_space, r_space, N_max_space, res_array)

def task2_2(new_dataset_name, N_space = [5e4, 1e5, 2e5], N_m = 1, m = 3, plot_measured_k_max = False):
    
    k_max_avg_array, k_max_std_array,res_array = N_scaling_degree_distribution(new_dataset_name, N_space, N_m=N_m, m=m, PS = 'RA', bin_scale = 1.3)
    k_max_analysis('RA', N_m, N_space, m, m, k_max_avg_array, k_max_std_array,res_array, plot_measured_k_max = plot_measured_k_max)

def task2_2_load(dataset_name, plot_measured_k_max = False):
    
    PS, N_m, bin_scale, m, r, N_max_space, k_max_avg_array, k_max_std_array, res_array = load_N_scaling_degree_distribution('RA_' + dataset_name, keep_descriptors = True)
    k_max_analysis(PS, N_m, N_max_space, m, r, k_max_avg_array, k_max_std_array, res_array, plot_measured_k_max = plot_measured_k_max)

def task3_1(new_dataset_name, m_space = [3, 6], r_space = [1, 2], N_max = [5e3, 1e4], N_m = 1):
    
    if type(N_max) != list:
        N_max = [N_max] * len(m_space)
    res_array = m_scaling_degree_distribution(new_dataset_name, m_space, r_space, N_m=N_m, PS = ['RA', 'PA'], N_max = N_max, bin_scale = 1.3)
    k_degree_distribution_analysis(['RA', 'PA'], N_m, m_space, r_space, N_max, res_array)

def task3_1_load(dataset_name):
    
    PS, N_m, bin_scale, m_space, r_space, N_max_space, res_array = load_m_scaling_degree_distribution('RA_PA_' + dataset_name, keep_descriptors = True)
    k_degree_distribution_analysis(PS, N_m, m_space, r_space, N_max_space, res_array)

def task3_2(new_dataset_name, N_space = [5e4, 1e5, 2e5], N_m = 1, m = 3, r=1, plot_measured_k_max = False):
    
    k_max_avg_array, k_max_std_array,res_array = N_scaling_degree_distribution(new_dataset_name, N_space, N_m=N_m, m=m, r=r, PS = ['RA', 'PA'], bin_scale = 1.3)
    k_max_analysis(['RA', 'PA'], N_m, N_space, m, r, k_max_avg_array, k_max_std_array,res_array, plot_measured_k_max = plot_measured_k_max)

def task3_2_load(dataset_name, plot_measured_k_max = False):
    
    PS, N_m, bin_scale, m, r, N_max_space, k_max_avg_array, k_max_std_array, res_array = load_N_scaling_degree_distribution('RA_PA_' + dataset_name, keep_descriptors = True)
    k_max_analysis(PS, N_m, N_max_space, m, r, k_max_avg_array, k_max_std_array, res_array, plot_measured_k_max = plot_measured_k_max)


#task1_3('1_3_MEGABIG', [1, 3], N_max = [2e5, 4e5])
#task1_4_expected_k_max()
#task1_4('fourth', [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5], N_m = 5)
#task1_4_load('COMBINED_COMBINED_first_second_third')

#task2_1('1_3_5_large', [1, 3, 5], N_max = [5e4, 1e5, 2e5])
#task2_2("1e5_first", [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5], N_m = 20)
#task2_2_load("1e5_first", plot_measured_k_max = True)

#task3_1("EVM_3", [9], [3], [2e5])
#task3_1_load("EVM_1")
#task3_2("EVM_3_1_1e5_third", [1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5], N_m=5, m=3, r=1)
task3_2_load("EVM_3_1_1e5", True)

#PS, N_m, bin_scale, m, r, N_max_space, k_max_avg_array, k_max_std_array, res_array = combine_datasets('EVM_3_1_1e5', 'EVM_3_1_1e5_second', 'N', PS = ['RA','PA'])
#k_max_analysis(PS, N_m, N_max_space, m, r, k_max_avg_array, k_max_std_array, res_array)


#EVM_jebak = BA_network(m = 3, probability_strategy = ['RA','PA'], r = 1)
#EVM_jebak.initial_graph('regular', N=10, param = EVM_jebak.m)
#EVM_jebak.step()
#print(EVM_jebak)

