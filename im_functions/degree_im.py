#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:29:52 2018

@author: abhishek.umrawal
"""

def degree_im(network, budget):
    budget = min(budget,len(network.nodes)) 
    return [x[0] for x in sorted(dict(network.out_degree).items(), key=lambda item:item[1],reverse=True)][:budget]