#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:39:24 2018

@author: abhishek.umrawal
"""

import networkx as nx
import numpy as np
import logging
import timeit

from im_functions.independent_cascade import independent_cascade
from im_functions.linear_threshold import linear_threshold

def true_influence(inpt):
    #print('working')
    start = timeit.default_timer()
    network, seed_set, diffusion_model, n_sim, spontaneous_prob, name_id = inpt
    
    nodes = list(nx.nodes(network))
    influence = 0
            
    for j in range(n_sim):
        spontaneously_infected = []
        
        if len(spontaneous_prob) != 0:
            
            for m in range(len(network)):
                if np.random.rand() < spontaneous_prob[m]:
                    spontaneously_infected.append(nodes[m])
                    
        
        if diffusion_model == "independent_cascade":
            layers = independent_cascade(network, list(set(spontaneously_infected + seed_set)))  
        
        elif diffusion_model == "linear_threshold":
            layers = linear_threshold(network, list(set(spontaneously_infected + seed_set)))    
        
        for k in range(len(layers)):
            influence = influence + len(layers[k])
            
    influence = influence/n_sim
    
    results = [seed_set,influence]
    
    end = timeit.default_timer()
    #logging.info(str(results)+' Time taken: '+str(round(end - start,2))+' seconds.')
    
    return results