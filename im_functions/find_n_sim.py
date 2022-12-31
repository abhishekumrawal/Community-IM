#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 17:11:52 2020

@author: abhishek.umrawal
"""

import random
import numpy as np
from im_functions.influence import influence
#from im_functions.degree_im import degree_im

def find_n_sim(network, diffusion_model, max_budget, threshold=0.001, num_samples=10):
    
    n_sims = []
    
    for i in range(num_samples):
        
        #nodes_subset = degree_im(network,min(100,len(network.nodes)))
        
        "finding 100 nodes that are closest to the avg degree in the network"
        out_degree_dict = dict(network.out_degree())
        avg_out_degree = np.mean(list(out_degree_dict.values()))
        for key in out_degree_dict.keys():
            out_degree_dict[key] = abs(out_degree_dict[key] -  avg_out_degree)
        nodes_subset = [x[0] for x in sorted(out_degree_dict.items(), 
                              key=lambda item:item[1],reverse=False)][:min(100,len(network.nodes))]
        
        "picking a sample seed set from nodes_subsets of size max_budget"
        sample_seed_set = random.sample(nodes_subset,max_budget)
        
        "finding the n_sim"
        margin = threshold
        n_sim = 0
        true_influence_old = 0
        
        while (margin >= threshold):
            n_sim += 1
            true_influence_new = true_influence_old + (influence(network, sample_seed_set, diffusion_model, []) - true_influence_old)/n_sim
            print(i,n_sim,true_influence_new)
            margin = abs(true_influence_new - true_influence_old)
            true_influence_old = true_influence_new
        #print(n_sim)
        n_sims.append(n_sim)
    
    return max(n_sims)