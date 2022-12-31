#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:01:00 2019

@author: abhishek.umrawal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:17:14 2019

@author: abhishek.umrawal
"""

"importing required user-defined modules"
from im_functions.weighted_network import weighted_network
from im_functions.celfpp_im import celfpp_im
from im_functions.ccelfpp1_im import ccelfpp1_im
from im_functions.genetic_im import genetic_im
from im_functions.ris_im import ris_im
from im_functions.cris_im import cris_im
from im_functions.cofim_im import cofim_im
from im_functions.heuristic_im import heuristic_im

def non_adaptive_im(inpt):
    
    network, weighting_scheme, algorithm, heuristic, max_budget, diffusion_model, n_sim, name_id, community_method, communities, community_size_threshold, is_graph_already_weighted = inpt
    
    if is_graph_already_weighted == False:
        "## adding weights to the unweighted network"
        network = weighted_network(network,method = weighting_scheme)
    
    if (algorithm == "celfpp"):
        best_seed_sets, exp_influences, runtime = celfpp_im(network, weighting_scheme, max_budget, diffusion_model, n_sim, all_upto_budget=True) 
            
    elif (algorithm == "ccelfpp1"):
        best_seed_sets, exp_influences,runtime = ccelfpp1_im(network, weighting_scheme, max_budget, diffusion_model, n_sim, community_method, communities, community_size_threshold, all_upto_budget=True)
    
    elif (algorithm == "genetic"):
        best_seed_sets, exp_influences, runtime = genetic_im(network, weighting_scheme, max_budget, diffusion_model, n_sim, all_upto_budget=True)
    
    elif (algorithm == "ris"):
        best_seed_sets, exp_influences,runtime = ris_im(network, weighting_scheme, max_budget, diffusion_model, n_sim, all_upto_budget=True)

    elif (algorithm == "cris"):
        best_seed_sets, exp_influences,runtime = cris_im(network, weighting_scheme, max_budget, diffusion_model, n_sim,  community_method, communities, community_size_threshold, all_upto_budget=True)

    elif (algorithm == "cofim"):
        best_seed_sets, exp_influences, runtime = cofim_im(network, weighting_scheme, max_budget, diffusion_model, n_sim, community_method, communities, community_size_threshold, all_upto_budget=True)

    elif (algorithm == "heuristic"):
        best_seed_sets, exp_influences,runtime = heuristic_im(network, weighting_scheme, heuristic, max_budget, diffusion_model, n_sim, all_upto_budget=True)
    return 