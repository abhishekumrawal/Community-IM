#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:03:00 2019

@author: abhishek.umrawal
"""

"impoting required built-in modules"
import numpy as np
import pandas as pd

def weighted_network(network, method):
    
    np.random.seed(0)
    if (method == "rn"):
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = np.random.rand(1)[0]

    elif (method == "tv"):
        TV = [.1,.01,.001]
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = np.random.choice(TV)

    elif (method == "wc"):
        edge_list = pd.DataFrame(list(network.edges)) 
        edge_list.columns = ['from','to']
           
        for i in range(len(edge_list)):
            u = edge_list['from'][i]
            v = edge_list['to'][i]
            network[u][v]['act_prob'] = 1/(network.in_degree(v))
                  
    elif (method == "qn"):
        edge_list = pd.DataFrame(list(network.edges)) 
        edge_list.columns = ['from','to']
           
        for i in range(len(edge_list)):
            u = edge_list['from'][i]
            v = edge_list['to'][i]
            network[u][v]['act_prob'] = 2/(network.in_degree(u) + network.in_degree(v))
             
    elif (method == "cn"):
        for edge in network.edges():
            network[edge[0]][edge[1]]['act_prob'] = 0.01
            
    "printing edge weights"
    if 0:
        for (u,v) in network.edges():
            print(u, v, network.in_degree(u), network.in_degree(v), network[u][v]['act_prob'])  
    
    "input for linear threshold model"        
    for edge in network.edges():
        network[edge[0]][edge[1]]['influence'] = network[edge[0]][edge[1]]['act_prob']
                     
    return  network
