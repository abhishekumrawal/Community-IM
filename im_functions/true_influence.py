#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:39:24 2018

@author: abhishek.umrawal
"""

import logging
import timeit

import networkx as nx
import numpy as np
from tqdm import tqdm

from im_functions.independent_cascade import independent_cascade
from im_functions.linear_threshold import linear_threshold


def true_influence(inpt):
    # print('working')
    start = timeit.default_timer()
    network, seed_set, diffusion_model, n_sim, spontaneous_prob, name_id = inpt

    nodes = list(nx.nodes(network))
    influence = 0

    if not network.is_directed():
        network = network.to_directed()

    for _ in tqdm(range(n_sim)):
        new_seeds = seed_set

        if len(spontaneous_prob) != 0:
            spontaneously_infected = []
            for m in range(len(network)):
                if np.random.rand() < spontaneous_prob[m]:
                    spontaneously_infected.append(nodes[m])

            new_seeds = list(set(spontaneously_infected + new_seeds))

        if diffusion_model == "independent_cascade":
            layers = independent_cascade(network, new_seeds)

        elif diffusion_model == "linear_threshold":
            layers = linear_threshold(network, new_seeds)

        for k in range(len(layers)):
            influence = influence + len(layers[k])

    influence = influence / n_sim

    results = [seed_set, influence]

    end = timeit.default_timer()
    # logging.info(str(results)+' Time taken: '+str(round(end - start,2))+' seconds.')

    return results
