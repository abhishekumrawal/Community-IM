#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:29:52 2018

@author: abhishek.umrawal
"""

import json
import logging
import os
import pickle
import random
import timeit

# importing required built-in modules"
import numpy as np

# from im_functions.ivgreedy_im import ivgreedy_im
from im_functions.degdiscount_im import degdiscount_im

# importing required user-defined modules"
from im_functions.degree_im import degree_im


def heuristic_im(
    network,
    weighting_scheme,
    heuristic,
    budget,
    diffusion_model,
    n_sim=100,
    all_upto_budget=True,
):
    # results folder
    results_folder = "results/results_" + diffusion_model + "_" + weighting_scheme

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

    # calling the chosen heuristic method
    if heuristic == "degree":
        start = timeit.default_timer()
        best_seed_set = degree_im(network, budget)
        end = timeit.default_timer()
    # elif heuristic == 'ivgreedy':
    #    start = timeit.default_timer()
    #    best_seed_set = ivgreedy_im(network, budget)
    #    end = timeit.default_timer()
    elif heuristic == "degdiscount":
        start = timeit.default_timer()
        best_seed_set = degdiscount_im(network, budget)
        end = timeit.default_timer()

    runtime = end - start
    logging.info(
        "Time taken by " + heuristic + " is " + str(round(runtime, 2)) + " seconds."
    )

    # saving runtime info to a text file
    runtime_info = {heuristic: runtime}
    fstr = results_folder_runtime_files + os.sep + "runtime_info_" + heuristic + ".txt"
    with open(fstr, "w") as f:
        f.write(json.dumps(runtime_info))

    # generating nested final best seed sets
    final_best_seed_sets = [best_seed_set[: k + 1] for k, _ in enumerate(best_seed_set)]

    # saving results
    if all_upto_budget:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": heuristic,
            "n_sim": n_sim,
            "network_name": network.name,
            "best_seed_set": [[None]] + final_best_seed_sets,
            "exp_influence": [0] + final_exp_influences,
        }

        fstr = (
            results_folder_pickle_files
            + os.sep
            + "output_"
            + heuristic
            + "__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f2:
            pickle.dump(results, f2)

        logging.info("The final solution is as follows.")
        logging.info(str([[None]] + final_best_seed_sets))
        logging.info(str([0] + final_exp_influences))

        end = timeit.default_timer()
        logging.info(
            "Total time taken by "
            + heuristic
            + " is "
            + str(round(end - start, 2))
            + " seconds."
        )

        return (
            [[None]] + final_best_seed_sets,
            [0] + final_exp_influences,
            runtime_info[heuristic],
        )

    else:
        final_best_seed_set = final_best_seed_sets[-1]
        final_exp_influence = 0

        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": heuristic,
            "n_sim": n_sim,
            "network_name": network.name,
            "best_seed_set": final_best_seed_set,
            "exp_influence": final_exp_influence,
        }

        fstr = (
            results_folder_pickle_files
            + os.sep
            + "output_"
            + heuristic
            + "__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f2:
            pickle.dump(results, f2)

        logging.info("The final solution is as follows.")
        logging.info(str(final_best_seed_set))
        logging.info(str(final_exp_influence))

        end = timeit.default_timer()
        logging.info(
            "Total time taken by "
            + heuristic
            + " is "
            + str(round(end - start, 2))
            + " seconds."
        )

        return final_best_seed_set, final_exp_influence, runtime_info[heuristic]
