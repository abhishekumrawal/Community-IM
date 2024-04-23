#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:49:19 2019

@author: abhishek.umrawal
"""


def main_evaluation(
    name_id,
    graph_type,
    max_budget,
    diffusion_models,
    is_graph_already_weighted,
    weighting_schemes,
    num_procs,
    interval,
    all_algorithms,
):
    "importing required built-in modules"
    import itertools
    import logging
    import os as os
    import pickle as pickle
    import timeit
    from multiprocessing import Pool

    import networkx as nx

    "importing required user-defined modules"
    from im_functions.true_influence import true_influence
    from im_functions.weighted_network import weighted_network

    "dropping celfpp from "
    all_algorithms = all_algorithms[1:]

    "start timer"
    start = timeit.default_timer()

    "reading the network"
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

    "relabeling the nodes as positive integers viz. 1,2,..."
    network = nx.convert_node_labels_to_integers(network, first_label=1)

    "adding weights if the network is unweighted"
    if not is_graph_already_weighted:
        network = weighted_network(network, method=weighting_schemes[0])

    "results folder"
    results_folder = (
        "results/results_" + diffusion_models[0] + "_" + weighting_schemes[0]
    )

    "creating log files folder within the results folder"
    results_folder_log_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "log_files"
    )
    if not os.path.exists(results_folder_log_files):
        os.makedirs(results_folder_log_files)

    "remove the log file from previous runs"
    if os.path.exists(results_folder_log_files + os.sep + "exp_influence.log"):
        os.remove(results_folder_log_files + os.sep + "exp_influence.log")

    "removing exisiting log handlers"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    "set up logging to file"
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-6s %(levelname)-6s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=results_folder_log_files + os.sep + "exp_influence.log",
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
        "I am running main_evaluation.py for " + network.name[1:] + " network."
    )
    logging.info("The network is " + graph_type + ".")
    logging.info(
        "The network has "
        + str(len(network.nodes))
        + " nodes and "
        + str(len(network.edges))
        + " edges."
    )
    logging.info("I am using " + weighting_schemes[0] + " weighting scheme.")

    "results folder path"
    results_folder = results_folder + os.sep + "results" + name_id

    "reading the existing exp_influences_dict or declaring empty dictionary of exp_influences"
    try:
        filename = "exp_influences_dict.pkl"
        filename_with_path = (
            results_folder + os.sep + "exp_influences" + os.sep + filename
        )
        with open(filename_with_path, "rb") as f:
            exp_influences_dict = pickle.load(f)
    except Exception:
        exp_influences_dict = {}
    exp_influences_dict["name_id"] = name_id

    "reading seed set and exp_influence pair from celfpp results and appending them to exp_influences_dict"
    max_budget = min(max_budget, len(network.nodes))
    filename = "output_celfpp__" + str(max_budget) + "__.pkl"
    filename_with_path = results_folder + os.sep + "pickle_files" + os.sep + filename
    if os.path.exists(filename_with_path):
        with open(filename_with_path, "rb") as f:
            results_celfpp = pickle.load(f)
        for i, seed_set in enumerate(results_celfpp["best_seed_set"][1:]):
            exp_influences_dict[tuple(sorted(seed_set))] = results_celfpp[
                "exp_influence"
            ][i + 1]

    "seeds sets for which the expected influences are already there"
    seed_sets_in_exp_influences_dict = [
        set(item)
        for item in list(exp_influences_dict.keys())
        if not isinstance(item, str)
    ]

    "reading the output files except celfpp and looking for unique seed sets from all of them"
    unique_seed_sets = []
    # print(all_algorithms)
    for algorithm in all_algorithms:
        # print(algorithm)
        filename = "output_" + algorithm + "__" + str(max_budget) + "__.pkl"
        filename_with_path = (
            results_folder + os.sep + "pickle_files" + os.sep + filename
        )
        if os.path.exists(filename_with_path):
            print(filename_with_path)
            with open(filename_with_path, "rb") as f:
                results_dict = pickle.load(f)
            for seed_set in results_dict["best_seed_set"][1:]:
                # print(seed_set)
                if set(seed_set) not in unique_seed_sets:
                    # print(seed_set)
                    unique_seed_sets.append(set(seed_set))

    "removing seed sets which are already in seed_sets_in_exp_influences_dict"
    unique_seed_sets = [
        item
        for item in unique_seed_sets
        if item not in seed_sets_in_exp_influences_dict
    ]

    "working only for the interval"
    unique_seed_sets = [
        item for item in unique_seed_sets if len(item) % interval == 0 or len(item) == 1
    ]

    try:
        unique_seed_sets.remove({None})
    except Exception:
        pass

    # for seed_set in unique_seed_sets:
    #    print(len(seed_set))

    # print(unique_seed_sets)

    logging.info(
        "There are " + str(len(unique_seed_sets)) + " unique seed sets to work with."
    )

    "diffusion model and number of simulations"
    diffusion_model = results_dict["diffusion_model"]
    n_sim = results_dict["n_sim"]
    exp_influences_dict["diffusion_model"] = diffusion_model
    exp_influences_dict["n_sim"] = n_sim

    "calculating the expected influence and saving in the exp_influences_dict"
    logging.info("I am using " + diffusion_model + " diffusion model.")
    logging.info("I am using " + str(n_sim) + " Monte-Carlo simulations.")

    "set to list for seed sets in unique_seed_sets"
    unique_seed_sets = [list(item) for item in unique_seed_sets]

    "create a list of all parameter lists, then use product"
    tmp = [[network], unique_seed_sets, [diffusion_model], [n_sim], [[]], [name_id]]
    inputs = itertools.product(*tmp)
    inputs = [tuple(i) for i in inputs]

    "parallelization"
    pool = Pool(processes=num_procs)
    exp_influences_list = list(pool.map(true_influence, inputs))
    pool.close()
    pool.join()

    "appending the [seed_set, exp_influence] pairs from exp_influences_list to exp_influences_dict"
    for [seed_set, exp_influence] in exp_influences_list:
        exp_influences_dict[tuple(sorted(set(seed_set)))] = exp_influence

    "saving the exp_influences_dict as a pickle file"
    filename = "exp_influences_dict.pkl"
    filename_with_path = results_folder + os.sep + "exp_influences" + os.sep + filename
    if not os.path.exists(results_folder + os.sep + "exp_influences"):
        os.makedirs(results_folder + os.sep + "exp_influences")
    with open(filename_with_path, "wb") as f:
        pickle.dump(exp_influences_dict, f)

    "adding exp influences to the original output files"
    for algorithm in all_algorithms:
        filename = "output_" + algorithm + "__" + str(max_budget) + "__.pkl"
        filename_with_path = (
            results_folder + os.sep + "pickle_files" + os.sep + filename
        )
        if os.path.exists(filename_with_path):
            "reading the file"
            with open(filename_with_path, "rb") as f:
                results_dict = pickle.load(f)
            "keep only the first entry i.e. 0 in case the other values already exist"
            results_dict["exp_influence"] = [results_dict["exp_influence"][0]]
            "adding the exp influences"
            for seed_set in results_dict["best_seed_set"][1:]:
                if tuple(sorted(seed_set)) in exp_influences_dict.keys():
                    results_dict["exp_influence"].append(
                        exp_influences_dict[tuple(sorted(seed_set))]
                    )
                else:
                    results_dict["exp_influence"].append(0)
            "saving the file"
            with open(filename_with_path, "wb") as f:
                pickle.dump(results_dict, f)

            # print(algorithm)
            # print(results_dict['exp_influence'])

    logging.info(
        "I finished running main_evaluation.py for " + network.name[1:] + " network."
    )
    logging.info("The network is " + graph_type + ".")
    logging.info(
        "The network has "
        + str(len(network.nodes))
        + " nodes and "
        + str(len(network.edges))
        + " edges."
    )
    logging.info("I used " + weighting_schemes[0] + " weighting scheme.")
    logging.info("I used " + diffusion_model + " diffusion model.")
    logging.info("I used " + str(n_sim) + " Monte-Carlo simulations.")

    "end timer"
    end = timeit.default_timer()
    logging.info(
        "Total time taken by main_exp_influence.py is "
        + str(round(end - start, 2))
        + " seconds."
    )
