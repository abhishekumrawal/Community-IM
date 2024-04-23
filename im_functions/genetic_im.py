#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: abhishek.umrawal
"""

import json
import logging
import os
import pickle
import random
import timeit
from multiprocessing import Pool

import networkx as nx
import numpy as np

from im_functions.true_influence import true_influence


def genetic_im(
    network, weighting_scheme, budget, diffusion_model, n_sim, all_upto_budget=True
):
    """
    Genetic Algorithm for finding the best seed set of a user-specified budget

    Inputs:
        - network is a networkx object
        - budget is the user-specified marketing budget which represents the no.
          of individuals to be given the freebies
        - diffusion model is either "independent_cascade" or "linear_threshold"
        - n_sim is the no. of simulations to be perfomed to estimate the expected influence
        - influence_dict is a dictionary of seed set and expected influence pairs to save computation
        - spontaneous_prob is a vector of spontaneous adoption probabiities for
          each node
        - pop_size is the no. of candidate solutions to be tried
        - lam is a parameter for the logistic policy
        - p_mut is the probability of mutation
        - crossover is a flag if you want crossover or not

    Outputs:
        - best_seed_set is a subset of the set of nodes with the cardinality
          same as the budget such that it maximizes the spread of marketing
        - max_influence is the value of maximum influence

    """
    results_folder = "results/results_" + diffusion_model + "_" + weighting_scheme

    np.random.seed(int(random.uniform(0, 1000000)))

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

    start = timeit.default_timer()
    print("pruning based on centrality measures")
    print("calculating betweenness")
    betweenness_list = list(nx.betweenness_centrality(network).values())
    print("calculating closeness")
    closeness_list = list(nx.closeness_centrality(network).values())
    # katz_list = list(nx.katz_centrality(network).values())

    # ranked_nodes = [x+1 for x in np.argsort([-sum(x) for x in zip(betweenness_list, closeness_list, katz_list)])]
    ranked_nodes = [
        x + 1
        for x in np.argsort([-sum(x) for x in zip(betweenness_list, closeness_list)])
    ]

    num_nodes = len(network.nodes)
    prop = 0.50

    nodes_after_pruning = ranked_nodes[: int(prop * num_nodes)]

    influence_dict = {}
    pop_size = 100
    # max_iter_multiplier = 100
    # lam = 1
    p_mut = 0.3
    crossover = True
    n_sim = 100
    budget_interval = 20

    if all_upto_budget:
        budgets = [1] + list(range(budget_interval, budget + 1, budget_interval))
    else:
        budgets = [budget]

    best_seed_sets = []
    max_influences = []
    runtime_info = {}
    for b in budgets:
        print("budget: " + str(b))
        # max iterations
        max_iter = 10  # max_iter_multiplier*budget
        # print(max_iter)

        # list of nodes
        nodes = nodes_after_pruning

        # empty output lists
        best_seed_set = []
        max_influence = []

        # number of vertices in the network
        n = len(nodes_after_pruning)

        # initializing a population of size pop_size
        A = np.zeros((pop_size, n))
        for i in range(len(A)):
            A[i][list(np.random.choice(n, b, replace=True))] = 1
        # print(A.sum(1))

        # initializing the fitness variable
        f = np.zeros(len(A))

        # initializing the selection probabilities
        prob = np.zeros(len(A))

        for t in range(max_iter):
            print(
                "max iteration: " + str(max_iter) + ", current iteration: " + str(t + 1)
            )

            # calcuating fitness for every candidate in the population
            # which is not already in influence_dict"
            # print('fitness calculation ...')

            inputs = []
            keys = []

            for i in range(len(A)):
                seed_set = [nodes[k] for k in list(np.where(A[i] == 1)[0])]
                if tuple(seed_set) not in influence_dict.keys():
                    inpt = [network, seed_set, diffusion_model, n_sim, [], network.name]
                    inputs.append(inpt)
                    keys.append(tuple(seed_set))
            pool = Pool(processes=os.cpu_count())
            influence_values_pairs = list(pool.map(true_influence, inputs))
            pool.close()
            pool.join()

            # updating the influence_dict for new keys from above"
            influence_values = [val_pair[1] for val_pair in influence_values_pairs]
            influence_dict_temp = dict(zip(keys, influence_values))
            influence_dict.update(influence_dict_temp)

            # looking up fitness for every candidate in the population"
            # in influence_dict"
            for i in range(len(A)):
                seed_set = [nodes[k] for k in list(np.where(A[i] == 1)[0])]
                f[i] = influence_dict[tuple(seed_set)]

            # finding current best seed set and the max influence
            best_seed_set = [
                nodes[k] for k in [j for j, x in enumerate(A[np.argmax(f)]) if x == 1]
            ]
            max_influence = influence_dict[tuple(best_seed_set)]

            # reproduction
            # prob = np.round(np.exp(lam * (f))/(np.sum(np.exp(lam * (f)))),20)
            sumf = sum(f)
            prob = [val / sumf for val in f]
            # print(prob)

            for i in range(len(A)):
                A[i] = A[np.random.choice(len(A), p=prob)]

            # crossover
            if crossover:
                # top two in A
                top_two = A[np.argsort([-x for x in f])[0:2]]

                # nodes in top two
                nodes_top_1 = [nodes[k] for k in list(np.where(top_two[0] == 1)[0])]
                nodes_top_2 = [nodes[k] for k in list(np.where(top_two[1] == 1)[0])]

                # merged and sorted nodes
                merged_and_sorted = sorted(nodes_top_1 + nodes_top_2)

                # offsprings
                offspring1 = [x for i, x in enumerate(merged_and_sorted) if i % 2 == 0]
                offspring2 = [x for i, x in enumerate(merged_and_sorted) if i % 2 != 0]

                # row for A for offspring 1
                A_offspring1 = [0] * n
                for node in offspring1:
                    loc = nodes_after_pruning.index(node)
                    A_offspring1[loc - 1] = 1

                # row for A for offspring 2
                A_offspring2 = [0] * n
                for node in offspring2:
                    loc = nodes_after_pruning.index(node)
                    A_offspring2[loc - 1] = 1

                # updating_population
                A[
                    np.random.choice(
                        np.argsort(f)[len(f) - 10 : len(f)], 2, replace=False
                    )
                ] = np.vstack((np.array(A_offspring1), np.array(A_offspring2)))

            # mutation
            for i in range(len(A)):
                if np.random.rand() < p_mut:
                    one_locs = list(np.where(A[i] == 1)[0])
                    zero_locs = list(np.where(A[i] == 0)[0])

                    A[i][np.random.choice(one_locs)] = 0
                    A[i][np.random.choice(zero_locs)] = 1

        best_seed_sets.append(best_seed_set)
        max_influences.append(max_influence)

        end = timeit.default_timer()
        runtime = end - start

        runtime_info[b] = runtime
        fstr = results_folder_runtime_files + os.sep + "runtime_info_genetic.txt"
        with open(fstr, "w") as f:
            f.write(json.dumps(runtime_info))

    best_seed_sets1 = [[None]] * budget
    for i, j in enumerate(budgets):
        best_seed_sets1[j - 1] = best_seed_sets[i]

    max_influences1 = [0] * budget
    for i, j in enumerate(budgets):
        max_influences1[j - 1] = max_influences[i]

    if all_upto_budget:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "genetic",
            "n_sim": n_sim,
            "best_seed_set": [[None]] + best_seed_sets1,
            "network_name": network.name,
            "exp_influence": [0] + max_influences1,
        }

        fstr = (
            results_folder_pickle_files + os.sep + "output_genetic__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f:
            pickle.dump(results, f)

        logging.info("The final solution is as follows.")
        logging.info(str([[None]] + best_seed_sets1))
        logging.info(str([0] + max_influences1))
        logging.info(
            "Total time taken by Genetic-IM is "
            + " "
            + str(round(runtime, 2))
            + " seconds."
        )

        return [[None]] + best_seed_sets1, [0] + max_influences1, runtime

    else:
        results = {
            "budget": budget,
            "diffusion_model": diffusion_model,
            "algorithm": "genetic",
            "n_sim": n_sim,
            "best_seed_set": best_seed_sets1[-1],
            "network_name": network.name,
            "exp_influence": max_influences1[-1],
        }

        fstr = (
            results_folder_pickle_files + os.sep + "output_genetic__%i__.pkl" % (budget)
        )
        with open(fstr, "wb") as f:
            pickle.dump(results, f)

        logging.info("The final solution is as follows.")
        logging.info(str(best_seed_sets1[-1]))
        logging.info(max_influences1[-1])
        logging.info(
            "Total time taken by Genetic-IM is "
            + " "
            + str(round(runtime, 2))
            + " seconds."
        )

        return best_seed_sets1[-1], max_influences1[-1], runtime
