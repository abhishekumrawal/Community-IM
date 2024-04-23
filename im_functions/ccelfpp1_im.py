#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:29:52 2018

@author: abhishek.umrawal
"""

# importing required built-in modules
import json
import logging
import os
import pickle
import random
import timeit

import numpy as np
from networkx.algorithms.community import is_partition, modularity, partition_quality

# importing required user-defined modules
from im_functions.celfpp_im import celfpp_im
from im_functions.community_detection import community_detection
from im_functions.progressive_budgeting import progressive_budgeting


def ccelfpp1_im(
    network,
    weighting_scheme,
    budget,
    diffusion_model,
    n_sim=100,
    community_method="louvain",
    communities=[],
    community_size_threshold=0.01,
    all_upto_budget=True,
):
    # start the timer
    start = timeit.default_timer()

    # results folder
    results_folder = "results/results_" + diffusion_model + "_" + weighting_scheme

    # initialize the run_time info dictionary
    runtime_info = {}

    # set a random seed
    np.random.seed(int(random.uniform(0, 1000000)))

    # set budget as number of nodes if the user gives a budget > number of nodes
    budget = min(budget, len(network.nodes))

    # initialize empty output lists
    final_best_seed_sets = []
    final_exp_influences = []

    # creating pickle files folder within the results folder
    results_folder_pickle_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "pickle_files"
    )
    if not os.path.exists(results_folder_pickle_files):
        os.makedirs(results_folder_pickle_files)

    # creating log files folder within the results folder
    results_folder_log_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "log_files"
    )
    if not os.path.exists(results_folder_log_files):
        os.makedirs(results_folder_log_files)

    # creating runtime files folder within the results folder
    results_folder_runtime_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "runtime_files"
    )
    if not os.path.exists(results_folder_runtime_files):
        os.makedirs(results_folder_runtime_files)

    # part 1: community detection
    if communities == []:
        logging.info(
            "Part 1: Detecting communities using " + community_method + " method."
        )
        startc = timeit.default_timer()
        communities = community_detection(network, community_method)
        endc = timeit.default_timer()
        runtime_info["community_detection"] = endc - startc
        logging.info(
            "Time taken in community detection is "
            + str(round(endc - startc, 2))
            + " seconds."
        )
    else:
        runtime_info["community_detection"] = 0
        logging.info("Time taken in community detection is " + str(0) + " seconds.")

    # community statistics
    num_communities = len(communities)
    sizes_communities = [len(community) for community in communities]
    is_valid_partition = is_partition(network, communities)
    coverage_score = partition_quality(network, communities)[0]
    # performance_score = performance(network, communities)
    modularity_score = modularity(network, communities, weight="act_prob")

    logging.info(
        "Number of communities detected/provided is " + str(num_communities) + "."
    )
    logging.info("Sizes of the communities are as follwows.")
    logging.info(str(sizes_communities))
    logging.info("Communities form a valid partition: " + str(is_valid_partition) + ".")
    logging.info("Coverage of the partition is " + str(coverage_score))
    # logging.info('Performance of the partition: ' + str(performance_score)+'.')
    logging.info("Modularity of the partition is " + str(modularity_score) + ".")

    # finding small communities and merging them together
    small_communities = [
        community
        for community in communities
        if len(community) < community_size_threshold * len(network.nodes)
    ]
    large_communities = [
        community for community in communities if community not in small_communities
    ]

    merged_small_communities = [
        item for sublist in small_communities for item in sublist
    ]

    if len(merged_small_communities) > 0:
        new_communities = large_communities + [merged_small_communities]
    else:
        new_communities = large_communities

    # print('Number of new communities after merging the small ones together is '+str(len(new_communities))+'.')
    logging.info(
        "Number of new communities detected/provided is "
        + str(len(new_communities))
        + "."
    )
    logging.info("Sizes of the new communities are as follwows.")
    logging.info(str([len(community) for community in new_communities]))
    logging.info(
        "New communities form a valid partition: "
        + str(is_partition(network, new_communities))
        + "."
    )
    logging.info(
        "Coverage of the new partition is "
        + str(partition_quality(network, new_communities)[0])
    )
    # logging.info('Performance of the new partition: ' + str(performance(network, new_communities))+'.')
    logging.info(
        "Modularity of the new partition is "
        + str(modularity(network, new_communities))
        + "."
    )

    # using new communities as the final communities for the next step
    communities = new_communities

    # part 2: celfpp within each community
    logging.info("Part 2: Solving influence maximization for each community.")
    all_communities_results = {}
    runtime_info["celfpp"] = 0
    for i, community in enumerate(communities):
        starti = timeit.default_timer()
        logging.info("Community-CELF++ is working on community " + str(i) + ".")
        sub_network = (
            network.subgraph(community).copy()
        )  # .copy() makes a subgraph with its own copy of the edge/node attributes
        sub_network.name = network.name + "_community_" + str(i)
        budget_for_greedy = min(len(sub_network.nodes), budget)
        best_seed_set, exp_influence, runtime = celfpp_im(
            sub_network,
            weighting_scheme,
            budget_for_greedy,
            diffusion_model,
            n_sim,
            all_upto_budget=True,
        )

        runtime_info["celfpp"] += runtime
        logging.info(str(best_seed_set))
        logging.info(str(exp_influence))
        all_communities_results[i] = [
            list(range(0, budget_for_greedy + 1)),
            best_seed_set,
            exp_influence,
        ]
        endi = timeit.default_timer()
        logging.info(
            "Time taken by Community-CELF++ for community "
            + str(i)
            + " is "
            + str(round(endi - starti, 2))
            + " seconds."
        )

    # print(all_communities_results)

    # part 3.1: progressive budgeting
    logging.info("Part 3.1: Solving the progressive budgeting problem.")
    startk = timeit.default_timer()
    final_best_seed_sets = progressive_budgeting(all_communities_results, budget)
    endk = timeit.default_timer()
    runtime_info["progressive_budgeting"] = endk - startk
    logging.info("The final best seed sets are as follows.")
    logging.info(str(final_best_seed_sets))
    logging.info(
        "Time taken in progressive budgeting is "
        + str(round(endk - startk, 2))
        + " seconds."
    )
    end = timeit.default_timer()
    logging.info(
        "Total time taken by Community-CELF++ until now is"
        + " "
        + str(round(end - start, 2))
        + " seconds."
    )

    # saving runtime info
    runtime_info["ccelfpp1"] = (
        runtime_info["community_detection"]
        + runtime_info["celfpp"]
        + runtime_info["progressive_budgeting"]
    )
    fstr = results_folder_runtime_files + os.sep + "runtime_info_ccelfpp1.txt"
    with open(fstr, "w") as f:
        f.write(json.dumps(runtime_info))

    # saving results
    if all_upto_budget:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "ccelfpp1",
            "n_sim": n_sim,
            "community_method": community_method,
            "communities": communities,
            "num_communities": num_communities,
            "sizes_communities": sizes_communities,
            "is_valid_partition": is_valid_partition,
            "coverage_score": coverage_score,
            "modularity_score": modularity_score,
            "network_name": network.name,
            "best_seed_set": [[None]] + final_best_seed_sets,
            "exp_influence": [0] + final_exp_influences,
        }

        fstr = (
            results_folder_pickle_files
            + os.sep
            + "output_ccelfpp1__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f2:
            pickle.dump(results, f2)

        logging.info("The final solution is as follows.")
        logging.info(str([[None]] + final_best_seed_sets))
        logging.info(str([0] + final_exp_influences))

        end = timeit.default_timer()
        logging.info(
            "Total time taken by Community-CELF++ is "
            + str(round(end - start, 2))
            + " seconds."
        )

        return (
            [[None]] + final_best_seed_sets,
            [0] + final_exp_influences,
            runtime_info["ccelfpp1"],
        )

    else:
        final_best_seed_set = final_best_seed_sets[-1]
        final_exp_influence = 0

        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "ccelfpp1",
            "n_sim": n_sim,
            "community_method": community_method,
            "communities": communities,
            "num_communities": num_communities,
            "sizes_communities": sizes_communities,
            "is_valid_partition": is_valid_partition,
            "coverage_score": coverage_score,
            "modularity_score": modularity_score,
            "network_name": network.name,
            "best_seed_set": final_best_seed_set,
            "exp_influence": final_exp_influence,
        }

        fstr = (
            results_folder_pickle_files
            + os.sep
            + "output_ccelfpp1__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f2:
            pickle.dump(results, f2)

        logging.info("The final solution is as follows.")
        logging.info(str(final_best_seed_set))
        logging.info(str(final_exp_influence))

        end = timeit.default_timer()
        logging.info(
            "Total time taken by Community-CELF++ is "
            + str(round(end - start, 2))
            + " seconds."
        )

        return final_best_seed_set, final_exp_influence, runtime_info["ccelfpp1"]
