#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 04:46:04 2021

@author: abhishek.umrawal
"""

# importing required user-defined modules
from main_functions.main_compilation import main_compilation
from main_functions.main_evaluation import main_evaluation
from main_functions.main_offline import main_offline
from main_functions.main_summarization import main_summarization

"""
"define ALL inputs as follows"
"""

"""inputs common to all main files"""

"network information"
name_id = "_facebook"  # network to work with and identifier to save the results
graph_type = "undirected"  # undirected, directed

"maximum budget"
max_budget = 100  # just provide the max seed set size

"algorithms to implement"
algorithms = [
    "celfpp",
    "ccelfpp1",
]  # ['heuristic','celfpp','ccelfpp1','genetic','cofim']
heuristics = []  # ['degree','degdiscount']


"diffusion models, weighting schemes and number of simulations"
diffusion_models = ["independent_cascade"]  # 'independent_cascade','linear_threshold'
is_graph_already_weighted = False  # False if the graph is not already weighted
weighting_schemes = ["wc"]  # 'tv' 'rn' 'wc'
n_sim = 1000  # number of diffusions

"community detection related inputs"
communities = []  # pre-defined communities

if "_sbm" in name_id or "_lfr" in name_id:
    with open("network_data/" + name_id[1:] + "_network_communities.txt") as f:
        lines = f.readlines()

    node_ids = []
    labels = []
    for line in lines:
        node_ids.append(int(line.split(" ")[0]))
        labels.append(int(line.split(" ")[1]))
    node_ids = [x + 1 for x in node_ids]
    labels = [x + 1 for x in labels]

    communities = []
    for val in list(set(labels)):
        community = []
        for i, label in enumerate(labels):
            if label == val:
                community.append(node_ids[i])
        communities.append(community)

community_methods = ["louvain"]  # louvain, label_propagation, girvan_newman
community_size_threshold = (
    0.01  # merging communities with nodes less than threshold % of all nodes
)

"""inputs specific to main2_exp_influence file and main3_compilation files"""
# import os; num_process = os.cpu_count
num_procs = 33  # number of processors for multiprocessing
interval = 5  # expected influence calculation interval

"""inputs specific to main4_run_times and main5_max_exp_influences files"""

all_algorithms = [
    "celfpp",
    "ccelfpp1",
    "genetic",
    "cofim",
    "degree",
    "degdiscount",
]  # all algorithms
all_name_ids = ["_facebook", "_bitcoin", "_wikipedia", "_epinions"]  # all networks

"""specify what all you need to do"""
run_algos = 0
evaluation = 1
compilation = 1
summarization = 1

"""
running different main functions sequentially for the inputs defined above"
"""
# if __name__ == '__main__':

if run_algos:
    "running different algorithms"
    main_offline(
        name_id,
        graph_type,
        max_budget,
        algorithms,
        heuristics,
        diffusion_models,
        is_graph_already_weighted,
        weighting_schemes,
        n_sim,
        communities,
        community_methods,
        community_size_threshold,
    )

if evaluation:
    "evaluating different algorithms"
    main_evaluation(
        name_id,
        graph_type,
        max_budget,
        diffusion_models,
        is_graph_already_weighted,
        weighting_schemes,
        num_procs,
        interval,
        all_algorithms,
    )

if compilation:
    "compiling and plotting results for different algorithms"
    main_compilation(
        name_id,
        graph_type,
        max_budget,
        diffusion_models,
        weighting_schemes,
        interval,
        all_algorithms,
    )
if summarization:
    "tabulating run-times and max expected influences for different algorithms all networks"
    main_summarization(
        max_budget,
        diffusion_models,
        weighting_schemes,
        interval,
        all_algorithms,
        all_name_ids,
    )
