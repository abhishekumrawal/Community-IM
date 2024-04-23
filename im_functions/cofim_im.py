#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:30:04 2019

@author: abhishek.umrawal
"""

import json
import logging
import os
import pickle
import random
import shutil
import timeit

import networkx as nx
import numpy as np
from networkx.algorithms.community import is_partition, modularity, partition_quality

from im_functions.community_detection import community_detection


def cofim_im(
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
    # results folder
    results_folder = "results/results_" + diffusion_model + "_" + weighting_scheme

    # initialize the run_time info dictionary
    runtime_info = {}

    # set a random seed
    np.random.seed(int(random.uniform(0, 1000000)))

    # creating a copy of the cofim_folder
    cofim_folder = "./cofim_code_release"

    if not os.path.exists(cofim_folder + "_copies"):
        os.makedirs(cofim_folder + "_copies")

    cofim_folder_new = (
        cofim_folder
        + "_copies"
        + os.sep
        + "cofim_"
        + network.name[1:]
        + "_"
        + diffusion_model
        + "_"
        + weighting_scheme
    )
    if os.path.exists(cofim_folder_new):
        shutil.rmtree(cofim_folder_new)
    shutil.copytree(cofim_folder, cofim_folder_new)

    # set budget as len(network.nodes) if the budget > len(network.nodes)
    budget = min(budget, len(network.nodes))

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

    # Generating input for CoFIM software
    ## saving the network as an adjacency list in a text file
    ## saving communities as one community per line in a text file
    ## saving the network as an adjacency list in a text file

    # saving network as an edge list in a text file
    network = nx.convert_node_labels_to_integers(network, first_label=0)
    nx.write_edgelist(
        network,
        cofim_folder_new + os.sep + network.name[1:] + ".txt",
        delimiter="\t",
        data=False,
    )

    # creating content for the first line as number of nodes and number of edges
    first_line = str(len(network.nodes())) + "\t" + str(len(network.edges()))

    # opening and reading the edge list file and prepending the first line content and then closing the file
    with open(cofim_folder_new + os.sep + network.name[1:] + ".txt", "r+") as f:
        existing_content = f.read()
        f.seek(0, 0)
        f.write(first_line.rstrip("\r\n") + "\n" + existing_content)

    ## saving communities as one community per line in a text file

    communities_relabeled = []
    for community in communities:
        community_relabeled = [val - 1 for val in community]
        communities_relabeled.append(community_relabeled)
    communities = communities_relabeled

    # community detection
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
    coverage_score, performance_score = partition_quality(network, communities)
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

    # saving communities as one community per line in a text file
    try:
        os.remove(cofim_folder_new + os.sep + network.name[1:] + "_com.txt")
    except OSError:
        pass
    with open(
        cofim_folder_new + os.sep + network.name[1:] + "_com.txt", "a"
    ) as outfile:
        for community in communities:
            commmunity_string = "\t".join(map(str, community))
            outfile.write("{}\n".format(commmunity_string))

    ### Running CoFIM software which is written in C and saving the outputs
    #### Changing directory to the Goyal's software folder, compiling the C codes by doing make

    os.chdir(cofim_folder_new)
    os.system("make")

    ##### Calling CoFIM
    start = timeit.default_timer()
    network_file = network.name[1:] + ".txt"
    network_com_file = network.name[1:] + "_com.txt"
    output_file = network.name[1:] + "_output.txt"
    best_seed_set_file = network.name[1:] + "_seeds.txt"
    try:
        os.remove(output_file)
    except OSError:
        pass
    os.system(
        ">>"
        + output_file
        + " ./CoFIM -data "
        + network_file
        + " -com "
        + network_com_file
        + " -gamma 3 -k "
        + str(budget)
    )
    end = timeit.default_timer()
    runtime = end - start

    # reading the output file
    output = (
        open(output_file).read().strip().split("time(s)")[1].strip().split("\n")[:-4]
    )

    # creating a list with the best seed set
    best_seed_set = [int(item.split("\t")[1]) for item in output]

    # writing best seed set to a text file
    seeds_textfile = open(best_seed_set_file, "w")
    for element in best_seed_set:
        seeds_textfile.write(str(element) + "\n")
    seeds_textfile.close()

    #### Changing the directory back to earlier
    os.chdir("..")
    os.chdir("..")

    ##### Saving runtime info to a text file
    runtime_info["cofimwc"] = runtime
    runtime_info["cofim"] = (
        runtime_info["community_detection"] + runtime_info["cofimwc"]
    )
    fstr = results_folder_runtime_files + os.sep + "runtime_info_cofim.txt"
    with open(fstr, "w") as f:
        f.write(json.dumps(runtime_info))

    # saving results
    best_seed_set = [item + 1 for item in best_seed_set]
    final_best_seed_sets = [best_seed_set[0 : k + 1] for k in range(0, budget)]
    final_exp_influences = []
    if all_upto_budget:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "cofim",
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
            results_folder_pickle_files + os.sep + "output_cofim__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f2:
            pickle.dump(results, f2)

        logging.info("The final solution is as follows.")
        logging.info(str([[None]] + final_best_seed_sets))
        logging.info(str([0] + final_exp_influences))

        end = timeit.default_timer()
        logging.info(
            "Total time taken by CoFIM is " + str(round(end - start, 2)) + " seconds."
        )

        return (
            [[None]] + final_best_seed_sets,
            [0] + final_exp_influences,
            runtime_info["cofim"],
        )

    else:
        final_best_seed_set = final_best_seed_sets[-1]
        final_exp_influence = 0

        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "cofim",
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
            results_folder_pickle_files + os.sep + "output_cofim__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f2:
            pickle.dump(results, f2)

        logging.info("The final solution is as follows.")
        logging.info(str(final_best_seed_set))
        logging.info(str(final_exp_influence))

        end = timeit.default_timer()
        logging.info(
            "Total time taken by CoFIM is " + str(round(end - start, 2)) + " seconds."
        )

        return final_best_seed_set, final_exp_influence, runtime_info["cofim"]
