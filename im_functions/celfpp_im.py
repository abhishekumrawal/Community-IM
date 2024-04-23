#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:30:04 2019

@author: abhishek.umrawal
"""

# importing required built-in modules"
import json
import logging
import os
import pickle
import random
import shutil
import timeit

import numpy as np
import pandas as pd


def celfpp_im(
    network,
    weighting_scheme,
    budget,
    diffusion_model,
    n_sim: int = 100,
    all_upto_budget: bool = True,
):
    results_folder = "results/results_" + diffusion_model + "_" + weighting_scheme

    np.random.seed(int(random.uniform(0, 1000000)))

    celfpp_folder = "./celf++_code_release"

    if not os.path.exists(celfpp_folder + "_copies"):
        os.makedirs(celfpp_folder + "_copies")

    celfpp_folder_new = (
        celfpp_folder
        + "_copies"
        + os.sep
        + "celf++_"
        + network.name[1:]
        + "_"
        + diffusion_model
        + "_"
        + weighting_scheme
    )
    if os.path.exists(celfpp_folder_new):
        shutil.rmtree(celfpp_folder_new)
    shutil.copytree(celfpp_folder, celfpp_folder_new)

    budget = min(budget, len(network.nodes))

    results_folder_pickle_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "pickle_files"
    )
    if not os.path.exists(results_folder_pickle_files):
        os.makedirs(results_folder_pickle_files)

    results_folder_log_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "log_files"
    )
    if not os.path.exists(results_folder_log_files):
        os.makedirs(results_folder_log_files)

    results_folder_runtime_files = (
        results_folder + os.sep + "results" + network.name + os.sep + "runtime_files"
    )
    if not os.path.exists(results_folder_runtime_files):
        os.makedirs(results_folder_runtime_files)

    act_prob = []
    for edge in network.edges():
        act_prob.append(network[edge[0]][edge[1]]["act_prob"])

    edge_list = pd.DataFrame(list(network.edges), columns=["from", "to"])
    graph = edge_list.copy(deep=True)
    graph["act_prob"] = act_prob
    graph.to_csv(
        celfpp_folder_new + "/datasets/graph.inf", sep=" ", index=False, header=True
    )

    input_file = celfpp_folder_new + "/config_test.txt"
    config_test = open(input_file).read().strip()

    if diffusion_model == "independent_cascade":
        config_test = " ".join(
            config_test.split(" ")[0:17]
            + ["IC" + "\n\nprobGraphFile"]
            + config_test.split(" ")[18:]
        )
    elif diffusion_model == "linear_threshold":
        config_test = " ".join(
            config_test.split(" ")[0:17]
            + ["LT" + "\n\nprobGraphFile"]
            + config_test.split(" ")[18:]
        )

    config_test = " ".join(
        config_test.split(" ")[0:19]
        + [celfpp_folder_new + "/datasets/graph.inf" + "\n\nmcruns"]
        + config_test.split(" ")[20:]
    )

    config_test = " ".join(
        config_test.split(" ")[0:21]
        + [str(n_sim) + "\n\noutdir"]
        + config_test.split(" ")[22:]
    )

    config_test = " ".join(
        config_test.split(" ")[0:23]
        + [celfpp_folder_new + "/output" + "\n\nbudget"]
        + config_test.split(" ")[24:]
    )

    config_test = " ".join(
        config_test.split(" ")[0:25]
        + [str(budget) + "\n\n#"]
        + config_test.split(" ")[26:]
    )

    with open(input_file, "w") as f:
        f.write(config_test)
        f.write("\n")

    os.chdir(celfpp_folder_new)
    os.system("make")
    os.chdir("..")
    os.chdir("..")

    start = timeit.default_timer()
    os.system(
        celfpp_folder_new
        + "/InfluenceModels -c "
        + celfpp_folder_new
        + "/config_test.txt"
    )
    end = timeit.default_timer()
    runtime = end - start

    # Saving runtime info to a text file
    runtime_info = {"celfpp": runtime}
    fstr = results_folder_runtime_files + os.sep + "runtime_info_celfpp.txt"
    with open(fstr, "w") as f:
        f.write(json.dumps(runtime_info))

    # Output filename for best seed set
    if diffusion_model == "independent_cascade":
        out_filename = celfpp_folder_new + "/output/IC_CelfPlus_Greedy.txt"
    elif diffusion_model == "linear_threshold":
        out_filename = celfpp_folder_new + "/output/LT_CelfPlus_Greedy.txt"

    # Getting the best seed set and exp influence
    best_seed_set_raw = [x.split(" ")[0] for x in open(out_filename).readlines()]
    exp_influence_raw = [x.split(" ")[1] for x in open(out_filename).readlines()]
    best_seed_set = [int(x) for x in best_seed_set_raw]
    exp_influence = [float(x) for x in exp_influence_raw]

    best_seed_set += random.sample(
        list(set(list(network.nodes)).difference(best_seed_set)),
        budget - len(best_seed_set),
    )
    exp_influence += [exp_influence[-1] for x in range(budget - len(exp_influence))]

    if all_upto_budget == True:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "celfpp",
            "n_sim": n_sim,
            "best_seed_set": [[None]]
            + [best_seed_set[: k + 1] for k, _ in enumerate(best_seed_set)],
            "network_name": network.name,
            "exp_influence": [0] + exp_influence,
        }

        fstr = (
            results_folder_pickle_files + os.sep + "output_celfpp__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f:
            pickle.dump(results, f)

        logging.info("The final solution is as follows.")
        logging.info(
            str(
                [[None]] + [best_seed_set[: k + 1] for k, _ in enumerate(best_seed_set)]
            )
        )
        logging.info(str([0] + exp_influence))
        logging.info(
            "Total time taken by CELF++ is "
            + " "
            + str(round(runtime, 2))
            + " seconds."
        )

        return (
            [[None]] + [best_seed_set[: k + 1] for k, _ in enumerate(best_seed_set)],
            [0] + exp_influence,
            runtime,
        )

    else:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "CELF++",
            "n_sim": n_sim,
            "best_seed_set": best_seed_set,
            "network_name": network.name,
            "exp_influence": exp_influence[-1],
        }

        fstr = (
            results_folder_pickle_files + os.sep + "output_celfpp__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f:
            pickle.dump(results, f)

        logging.info("The final solution is as follows.")
        logging.info(str(best_seed_set))
        logging.info(exp_influence[-1])
        logging.info(
            "Total time taken by CELF++ is "
            + " "
            + str(round(runtime, 2))
            + " seconds."
        )

        return best_seed_set, exp_influence[-1], runtime
