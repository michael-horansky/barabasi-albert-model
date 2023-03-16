
import numpy as np
import random as rd
import scipy as sp

import matplotlib.pyplot as plt
import time

from decimal import Decimal

from logbin_2020_CITE_THIS import *

# MATH FUNCTIONS
def weighted_sample(weights, sample_size, Z = -1):
    if sample_size > len(weights):
        return(-1)
    if Z == -1:
        Z = 0
        for w in weights:
            Z += w
    available_indices = list(np.arange(len(weights), dtype=np.int64))
    #print(available_indices)
    Z = int(Z)
    
    res = []
    for m in range(sample_size):
        dice_throw = rd.randrange(Z)
        #print("DICE", dice_throw)
        score_count = 0
        for i in range(len(available_indices)):
            score_count += weights[available_indices[i]]
            #print("score", score_count)
            if score_count > dice_throw:
                Z -= weights[available_indices[i]]
                res.append(available_indices.pop(i))
                break
    return(res)



class BA_network():
    
    # ------------ CONSTRUCTORS, DESTRUCTORS, DESCRIPTORS -----------------
    
    def reset(self):
        # We use the ADJACENCY LIST representation
        # adjacency_list[vertex index][neighbour index]
        # this is sane because in the BA model the graph is sparse
        self.adjacency_list = []
        self.t = 0
        
        # The probability is usually fully given by adjacency_list, but we trade some space off for speed
        # by storing a separate array unnorm_prob of size N, which keeps track of the weight of each node.
        # p_i = unnorm_prob(i) / Z; Z = sum_j unnorm_prob(j)
        self.unnorm_prob = []
        self.Z = 0
    
    def __init__(self, m = 3, probability_strategy = 'PA', r = 1):
        # Creates an empty network 
        
        self.m = m
        self.reset()
        
        # probability strategy: 'PA' = preferential attachement, 'RA' = random attachement
        #                       [PS1, PS2] = EVM with new vertex strategy PS1 and existing vertices strategy PS2
        self.probability_strategy = probability_strategy
        self.r = r
    
    def vertex_descriptor(self, i):
        adjacency_row_separator = ', '
        return(f"{i+1}: w = {self.unnorm_prob[i]}; [{adjacency_row_separator.join(str(item+1) for item in self.adjacency_list[i])}]")
        #return(f"{i+1}: w = {self.unnorm_prob[i]}; {self.adjacency_list[i]}")
    
    def __repr__(self):
        adjacency_list_separator = '\n  '
        return(f"m = {self.m}\nStrategy = {self.probability_strategy}\nZ={self.Z}\nAdjacency list =\n  {adjacency_list_separator.join(self.vertex_descriptor(i) for i in range(len(self.adjacency_list)))}")
        #return(f"L={self.L},p={self.p},h=[{','.join(str(item) for item in self.h)}],z_th=[{','.join(str(item) for item in self.z_th)}]")
    def __str__(self):
        adjacency_list_separator = '\n  '
        return(f"m = {self.m}\nStrategy = {self.probability_strategy}\nZ={self.Z}\nAdjacency list =\n  {adjacency_list_separator.join(self.vertex_descriptor(i) for i in range(len(self.adjacency_list)))}")
        #return(f"Oslo-model lattice; L = {self.L}, p = {self.p}\n{self.print_lattice_state()}")
    
    # ------------------ OPERATIVE FUNCTIONS ----------------------
    
    # --------------- Memory manipulation / Access
    def edge_exists(self, i, j):
        return(j in self.adjacency_list[i])
    
    
    def add_vertex(self):
        # Adds a new vertex, returns its index
        self.adjacency_list.append([])
        
        # Calculate its probability weight depending on the strategy
        if self.probability_strategy == 'PA':
            new_mode_weight = 0.0
        elif self.probability_strategy == 'RA':
            new_mode_weight = 1.0
        
        self.unnorm_prob.append(new_mode_weight)
        self.Z += new_mode_weight
        return(len(self.adjacency_list)-1)
        
    
    def add_edge(self, i, j):
        # Add an undirected edge between vertices with indices i and j
        self.adjacency_list[i].append(j)
        self.adjacency_list[j].append(i)
        
        # Update the probability weights if necessary
        if self.probability_strategy == 'PA':
            self.unnorm_prob[i] += 1.0
            self.unnorm_prob[j] += 1.0
            self.Z += 2.0
    
    
    # --------------- Initial graph generators
    def reset_to_complete_graph(self, N):
        self.reset()
        for i in range(N):
            self.add_vertex()
            for j in range(i):
                self.add_edge(i, j)
    def reset_to_ER_A_graph(self, N, p):
        self.reset()
        for i in range(N):
            self.add_vertex()
            for j in range(i):
                if rd.random() < p:
                    self.add_edge(i, j)
    def reset_to_ER_B_graph(self, N, E):
        self.reset()
        possible_edges = []
        for i in range(N):
            self.add_vertex()
            for j in range(i):
                possible_edges.append([i, j])
        # select E edges from the possible ones:
        actual_edges = rd.sample(possible_edges, E)
        for edge in actual_edges:
            self.add_edge(edge[0], edge[1])
    
    def reset_to_regular_graph(self, N, k):
        self.reset()
        incomplete_vertices = []
        for i in range(N):
            self.add_vertex()
            for j in range(k):
                incomplete_vertices.append(i)
        # no self-loops, no double edges
        # TODO make this more effective if possible (check McKay and Wormald)
        while(True):
            success = True
            while(len(incomplete_vertices) > 0 and success):
                # pick a random vertex
                a_i = rd.randrange(len(incomplete_vertices))
                a = incomplete_vertices.pop(a_i)
                
                # compile the list of suitable target vertex indices
                # For each target vertex, take the index of the first instance only to make this completely unbiased
                suitable_vertex_indices = []
                considered_targets = []
                for index in range(len(incomplete_vertices)):
                    if incomplete_vertices[index] != a and not self.edge_exists(a, incomplete_vertices[index]) and not incomplete_vertices[index] in considered_targets:
                        suitable_vertex_indices.append(index)
                        considered_targets.append(incomplete_vertices[index])
                
                #print(f"{a} : {considered_targets}")
                
                if len(suitable_vertex_indices) == 0:
                    success = False
                    break
                else:
                    b_i = suitable_vertex_indices[rd.randrange(len(suitable_vertex_indices))]
                b = incomplete_vertices.pop(b_i)
                self.add_edge(a, b)
            if success:
                break
            else:
                print("  Random regular graph generator hiccup...")
                self.reset()
                incomplete_vertices = []
                for i in range(N):
                    self.add_vertex()
                    for j in range(k):
                        incomplete_vertices.append(i)
    
    def initial_graph(self, strategy = 'complete', N = 10, param = 30, print_log = True):
        # a wrapper method for several functions generating the initial graph G_0
        if strategy.lower() in ['c', 'complete']:
            # Generates a complete graph of size N
            if print_log:
                print(f"Initialising a complete graph with {N} vertices...")
            self.reset_to_complete_graph(N)
        
        if strategy.lower() in ['e', 'er', 'erdos', 'ea', 'era', 'erdosa', 'e a', 'er a', 'erdos a', 'a']:
            # Generates an ER random graph type A of size N with p = param
            if print_log:
                print(f"Initialising an ER random graph type A with {N} vertices, p = {param}...")
            self.reset_to_ER_A_graph(N, param)
        
        if strategy.lower() in ['eb', 'erb', 'erdosb', 'e b', 'er b', 'erdos b', 'b']:
            # Generates an ER random graph type B of size N with E = param
            if print_log:
                print(f"Initialising an ER random graph type B with {N} vertices and {param} edges...")
            self.reset_to_ER_B_graph(N, param)
        
        if strategy.lower() in ['r', 'regular']:
            # Generates a random regular graph of size N, vertex degree = param
            if print_log:
                print(f"Initialising a random regular graph with {N} vertices, each of degree {param}...")
            if (N * param) % 2 == 1:
                # this is unresolvable
                print(f"N*k must be even... initialising a complete graph with {N} vertices")
                self.reset_to_complete_graph(N)
            else:
                self.reset_to_regular_graph(N, param)
    
    # ----------------- Model functions
    
    def step(self):
        # A single step (time increment) in the BA model
        self.t += 1
        
        #First we pick m vertices which will form an edge with the new vertex
        target_vertices = weighted_sample(self.unnorm_prob, self.m, self.Z)
        # Now we add the new vertex and create m edges
        new_i = self.add_vertex()
        for new_edge in range(self.m):
            self.add_edge(new_i, target_vertices[new_edge])
    
    def simulate(self, N_max, N_max_ultimate = -1, start_time = -1):
        # Will drive the model until the number of vertices reaches N_max
        # N_max_ultimate is the value of N_max which is used to calculate percentage done. If left unassigned, it will equal N_max
        # start_time is the system time at which the simulation begun - useful when N_max_ultimate != -1
        if N_max_ultimate == -1:
            N_max_ultimate = N_max
        
        delta_t = N_max_ultimate - len(self.adjacency_list)
        t_start = self.t
        if start_time == -1:
            start_time = time.time()
        progress_percentage = 0
        # TODO make the ETF smarter by considering each step has O(N)
        
        # We consider the total "time tokens": T = sum_N O_step(N) = N(0) + N(1) + ... + N_max = (N_max * (N_max + 1) - (N(0) - 1) * N(0)) / 2
        # Each step we count how many time tokens we accumulated and determine the percentage
        time_tokens_total = N_max_ultimate * (N_max_ultimate - 1.0) / 2.0
        time_tokens_accumulated = len(self.adjacency_list) * (len(self.adjacency_list) - 1.0) / 2.0
        
        while(len(self.adjacency_list) < N_max):
            time_tokens_accumulated += len(self.adjacency_list)
            if np.floor(time_tokens_accumulated / time_tokens_total * 100) > progress_percentage:
                progress_percentage = np.floor(time_tokens_accumulated / time_tokens_total * 100)
                #print("Analysis in progress: " + str(progress_percentage) + "%", end='\r')
                print("Analysis in progress: " + str(progress_percentage) + "%; est. time of finish: " + time.strftime("%H:%M:%S", time.localtime( (time.time()-start_time) * 100 / progress_percentage + start_time )), end='\r')
            self.step()
        if N_max == N_max_ultimate:
            print("Analysis done.                                                     ") #this is SUCH an ugly solution.
        #return(start_time)
    
    # ----------------- Analysis of self
    
    def degree_distribution(self, uniform_bins = False, uniform_bin_width = 1):
        degree_array = []
        for item in self.adjacency_list:
            degree_array.append(len(item))
        
        return(degree_array)
        """if uniform_bins:
            smallest_degree = min(degree_array)
            largest_degree = max(degree_array)
            bin_edges = np.arange(smallest_degree, largest_degree + 1, uniform_bin_width) - 0.5
            return(np.histogram(degree_array, bins = bin_edges))
        #else do logbinning
        return(logbin(degree_array))"""
    
    # ---------------- M TEST WRAPPER METHODS ---------------
    
    # For these, assume they will be called immediately after initialising the instance = you have to do EVERYTHING inside
    
    def get_degree_distribution(self, N_max = 1e4):
        self.initial_graph(strategy = 'r', N = 10, param = self.m)
        
        if type(N_max) != list:
            self.simulate(N_max)
            return(self.degree_distribution())
        else:
            degree_distribution_list = []
            first_start_time = time.time()
            for N_max_val in N_max:
                self.simulate(N_max_val, N_max_ultimate = N_max[-1], start_time = first_start_time)
                degree_distribution_list.append(self.degree_distribution())
            return(degree_distribution_list)
    
    def get_binned_degree_distribution(self, N_max = 1e4, bin_scale = 1.3):
        degree_distribution = self.get_degree_distribution(N_max)
        x, y_PDF, y_counts, binedges = logbin(degree_distribution, scale = bin_scale, x_min = self.m)
        

# ---------------- Existing Vertices Model ---------------------------

class EVM_network(BA_network):
    
    def reset(self):
        self.adjacency_list = []
        self.t = 0
        
        # We keep two unnorm probs and two Zs - one for new vertex, second for existing vertices
        self.unnorm_prob1 = []
        self.unnorm_prob2 = []
        self.Z1 = 0
        self.Z2 = 0
    def __init__(self, m = 3, r = 1, PS1='RA', PS2='PA'):
        # PS1 used for the new vertex
        # PS2 used for existing vertices
        self.m = m
        self.r = r
        self.reset()
        
        self.PS1 = PS1
        self.PS2 = PS2
        # so that we don't have to redefine descriptors
        self.probability_strategy = f"EVM with PS1 = '{PS1}', PS2 = '{PS2}'"
    
    
    # ------------------ OPERATIVE FUNCTIONS ----------------------
    
    def add_vertex(self):
        # Adds a new vertex, returns its index
        self.adjacency_list.append([])
        
        # Calculate its probability weight depending on the strategy
        if self.probability_strategy == 'PA':
            new_mode_weight = 0.0
        elif self.probability_strategy == 'RA':
            new_mode_weight = 1.0
        
        self.unnorm_prob.append(new_mode_weight)
        self.Z += new_mode_weight
        return(len(self.adjacency_list)-1)
        
    
    def add_edge(self, i, j):
        # Add an undirected edge between vertices with indices i and j
        self.adjacency_list[i].append(j)
        self.adjacency_list[j].append(i)
        
        # Update the probability weights if necessary
        if self.probability_strategy == 'PA':
            self.unnorm_prob[i] += 1.0
            self.unnorm_prob[j] += 1.0
            self.Z += 2.0

