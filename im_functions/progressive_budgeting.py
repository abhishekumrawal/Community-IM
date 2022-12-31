#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 04:33:59 2020

@author: abhishek.umrawal
"""
"importing required built-in modules"
import  numpy as np
from collections import defaultdict

def progressive_budgeting(all_communities_results, budget):
    all_communities_best_seed_sets = {}
    for key in all_communities_results.keys():
        all_communities_best_seed_sets[key] = all_communities_results[key][1:][0]
    
    all_communities_exp_influences = {}
    for key in all_communities_results.keys():
        all_communities_exp_influences[key] = all_communities_results[key][1:][1]
        
    all_communities_marginal_gains = {}
    for key in all_communities_exp_influences.keys():
        all_communities_marginal_gains[key]= [all_communities_exp_influences[key][i+1] - all_communities_exp_influences[key][i] \
                                              for i,_ in enumerate(all_communities_exp_influences[key]) \
                                                  if i+1 < len(all_communities_exp_influences[key])]
    
    all_communities_budget_allocation = defaultdict(int)
    final_best_seed_sets = []
    for k in range(1,budget+1): 
        current_communities = [key for key in all_communities_marginal_gains.keys()]     
        current_marginal_gains = [all_communities_marginal_gains[key][0] for key in all_communities_marginal_gains.keys()]
        chosen_community = current_communities[np.argmax(current_marginal_gains)]
        all_communities_budget_allocation[chosen_community] +=1
        all_communities_marginal_gains[chosen_community].remove(all_communities_marginal_gains[chosen_community][0])
        
        for key in list(all_communities_marginal_gains):
            if all_communities_marginal_gains[key] == []:
                del all_communities_marginal_gains[key]
                
        final_best_seed_set = []
        for key in all_communities_budget_allocation.keys():
            final_best_seed_set+= all_communities_best_seed_sets[key][all_communities_budget_allocation[key]]
        final_best_seed_sets.append(final_best_seed_set) 
            
    return final_best_seed_sets     