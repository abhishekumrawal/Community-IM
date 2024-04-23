#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:17:14 2019

@author: abhishek.umrawal
"""

# importing required built-in modules
import itertools
import logging
import os
import pickle
import shutil
import timeit

import networkx as nx
from networkx.classes.function import density

# importing required user-defined modules
from im_functions.non_adaptive_im import non_adaptive_im


def main_offline(
    name_id,
    graph_type,
    max_budget,
    algorithms,
    heuristics: list[str],
    diffusion_models,
    is_graph_already_weighted,
    weighting_schemes,
    n_sim,
    communities: list[list[int]],
    community_methods,
    community_size_threshold,
):
    # debugging functionality
    # clear : to clear the console
    # reset: to clear all variables
    # %run filename.py to run from console

    # DEBUG_MODE = False
    # import pdb

    # if DEBUG_MODE:
    #    pdb.set_trace()
    # else:
    #    pdb.set_trace = lambda: 1

    # important debugging tips
    # if don't specify pdb.set_trace() here then do python -m pdb <filename.py> from terminal
    # type c to continue running until next breakpoint()
    # type a to see arguments with their values for the current function
    # type print(<var_name>) to a print any variable created so far or
    # type p <var_name> for regular printing
    # type pp <var_name> for pretty printing
    # type !<var_name> to create a new variable
    # run
    # type q to quit -- all variables made until then would be saved
    # type pdb.set_trace = lambda: 1 to neutralize pdb for the entire session

    # start time
    start = timeit.default_timer()

    # adding '' to heuristics for celfpp and celfpp1
    heuristics = [""] + heuristics

    # reading the network
    if "florentine" in name_id:
        network = nx.florentine_families_graph()
        network.name = name_id
        network = network.to_directed()

    elif "florida" in name_id:
        network = nx.read_weighted_edgelist(
            "network_data/florida_network.csv",
            delimiter=",",
            create_using=nx.DiGraph(),
            nodetype=int,
        )
        nx.set_edge_attributes(
            network, values=nx.get_edge_attributes(network, "weight"), name="act_prob"
        )
        network.name = name_id

    elif graph_type == "directed":
        network = nx.read_edgelist(
            "network_data/" + name_id[1:] + "_network.txt",
            create_using=nx.DiGraph(),
            nodetype=int,
        )
        network.name = name_id

    elif graph_type == "undirected":
        network = nx.read_edgelist(
            "network_data/" + name_id[1:] + "_network.txt",
            create_using=nx.Graph(),
            nodetype=int,
        )
        network.name = name_id
        network = network.to_directed()

    # relabeling the nodes as positive integers viz. 1,2,...
    network = nx.convert_node_labels_to_integers(network, first_label=1)

    print("Network is connected: " + str(nx.is_strongly_connected(network)))

    "looking up the n_sim -- unused block"
    if 0:
        try:
            with open("network_data/n_sim_data.pkl", "rb") as f:
                n_sims = pickle.load(f)

            n_sim = n_sims[name_id]
        except Exception:
            pass

    "create a list of all parameter lists, then use product"
    tmp = [
        [network],
        weighting_schemes,
        algorithms,
        heuristics,
        [max_budget],
        diffusion_models,
    ]
    tmp += [[n_sim], [name_id]]
    tmp += [
        community_methods,
        [communities],
        [community_size_threshold],
        [is_graph_already_weighted],
    ]
    inputs = [tuple(i) for i in itertools.product(*tmp)]

    "removing redundant/unncessary inputs"
    inputs1 = []
    for inpt in inputs:
        if (
            inpt[2] in ["celfpp", "ccelfpp1", "genetic", "ris", "cris", "cofim"]
            and inpt[3] == ""
        ):
            inputs1.append(inpt)
        elif inpt[2] in ["heuristic"] and inpt[3] in heuristics[1:]:
            inputs1.append(inpt)

    "results folder"
    results_folder = (
        "results/results_" + diffusion_models[0] + "_" + weighting_schemes[0]
    )

    "call non_adaptive_im for all input combinations -- no parallelization"
    "python doesn't allow daemon processes to have children"
    "there is already parallelization in greedy_im and hence in cgreedy_im"
    for inpt in inputs1:
        "creating log files folder within the results folder"
        results_folder_log_files = (
            results_folder + os.sep + "results" + network.name + os.sep + "log_files"
        )
        if not os.path.exists(results_folder_log_files):
            os.makedirs(results_folder_log_files)

        "remove the log file from previous runs"
        if os.path.exists(
            results_folder_log_files
            + os.sep
            + "log_"
            + str(inpt[2])
            + "_"
            + str(inpt[3])
            + ".log"
        ):
            os.remove(
                results_folder_log_files
                + os.sep
                + "log_"
                + str(inpt[2])
                + "_"
                + str(inpt[3])
                + ".log"
            )

        "removing exisiting log handlers"
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        "set up logging to file"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)-6s %(levelname)-6s %(message)s",
            datefmt="%m-%d %H:%M",
            filename=results_folder_log_files
            + os.sep
            + str(inpt[2])
            + "_"
            + str(inpt[3])
            + ".log",
            filemode="w",
        )

        "define a Handler which writes INFO messages or higher to the sys.stderr"
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        "set a format which is simpler for console use"
        formatter = logging.Formatter("%(name)-6s: %(levelname)-6s %(message)s")

        "tell the handler to use this format"
        console.setFormatter(formatter)

        "add the handler to the root logger"
        logging.getLogger().addHandler(console)

        "Now, we can log to the root logger, or any other logger. First the root..."
        logging.info(
            "I am running "
            + inpt[2]
            + "_im"
            + " upto budget "
            + str(inpt[4])
            + " for "
            + name_id[1:]
            + " network."
        )
        logging.info("The network is " + graph_type + ".")
        logging.info(
            "The network has "
            + str(len(network.nodes))
            + " nodes and "
            + str(len(network.edges))
            + " edges."
        )
        logging.info("The network density is " + str(density(network)) + ".")
        logging.info(
            "I am using " + community_methods[0] + " community detection method."
        )
        logging.info("I am using " + weighting_schemes[0] + " weighting scheme.")
        logging.info("I am using " + diffusion_models[0] + " diffusion model.")
        logging.info("I am using " + str(n_sim) + " Monte-Carlo simulations.")

        "calling non_adaptive_im"
        non_adaptive_im(inpt)

        logging.info(
            "I finished running "
            + inpt[2]
            + "_im"
            + " upto budget "
            + str(inpt[4])
            + " for "
            + name_id[1:]
            + " network."
        )
        logging.info("The network is " + graph_type + ".")
        logging.info(
            "The network has "
            + str(len(network.nodes))
            + " nodes and "
            + str(len(network.edges))
            + " edges."
        )
        logging.info("I used " + community_methods[0] + " community detection method.")
        logging.info("I used " + weighting_schemes[0] + " weighting scheme.")
        logging.info("I used " + diffusion_models[0] + " diffusion model.")
        logging.info("I used " + str(n_sim) + " Monte-Carlo simulations.")

    if "ccelfpp1" in algorithms:
        "moving community wise results into a folder named all_community_results within results+network.name folder"
        if not os.path.exists(
            "."
            + os.sep
            + results_folder
            + os.sep
            + "results"
            + network.name
            + os.sep
            + "all_community_results"
        ):
            os.makedirs(
                "."
                + os.sep
                + results_folder
                + os.sep
                + "results"
                + network.name
                + os.sep
                + "all_community_results"
            )
        for folder_name in sorted(os.listdir("." + os.sep + results_folder)):
            if network.name + "_community" in folder_name:
                # print(folder_name)
                shutil.move(
                    "." + os.sep + results_folder + os.sep + folder_name,
                    "."
                    + os.sep
                    + results_folder
                    + os.sep
                    + "results"
                    + network.name
                    + os.sep
                    + "all_community_results",
                )

    "end time"
    end = timeit.default_timer()

    "time taken"
    logging.info("Total time taken is " + str(round(end - start, 4)) + " seconds.")
