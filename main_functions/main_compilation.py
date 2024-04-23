#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:49:19 2019

@author: abhishek.umrawal
"""


def main_compilation(
    name_id,
    graph_type,
    max_budget,
    diffusion_models,
    weighting_schemes,
    interval,
    all_algorithms,
):
    "importing required built-in modules"
    import os as os
    import pickle as pickle

    # matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams["figure.figsize"] = [8, 6]
    plt.rcParams.update({"font.size": 12})

    "chosen algorithms"
    if "independent_cascade" in diffusion_models:
        chosen_algorithm_indices = [0, 1, 2, 3, 4, 5]
    else:
        chosen_algorithm_indices = [0, 1, 2, 3, 4, 5]

    "results folder"
    results_folder = (
        "results/results_" + diffusion_models[0] + "_" + weighting_schemes[0]
    )
    results_folder = results_folder + os.sep + "results" + name_id

    "looking up for nested non-adaptive methods"
    chosen_algorithms = [
        item for i, item in enumerate(all_algorithms) if i in chosen_algorithm_indices
    ]
    output_dict = {}
    results_dicts = {}
    for algorithm in chosen_algorithms:
        filename = "output_" + algorithm + "__" + str(max_budget) + "__.pkl"
        filename_with_path = (
            results_folder + os.sep + "pickle_files" + os.sep + filename
        )
        if os.path.exists(filename_with_path):
            with open(filename_with_path, "rb") as f:
                results_dict = pickle.load(f)
            results_dicts[algorithm] = results_dict
            best_seed_sets = results_dict["best_seed_set"]
            exp_influences = results_dict["exp_influence"]

            # print(algorithm)
            # print(best_seed_sets)
            # print(exp_influences)

            if len(exp_influences) > 1:
                output_dict[algorithm + "_best_seed_sets"] = best_seed_sets
                output_dict[algorithm + "_exp_influences"] = exp_influences

    # print(output_dict['genetic_exp_influences'])
    # ""
    # budgets = [0,1] + list(range(20,max_budget+1,20))
    # print(budgets)
    # best_seed_sets1 = [[None]]*(max_budget+1)
    # for i,j in enumerate(budgets):
    #     best_seed_sets1[j] = output_dict['genetic_best_seed_sets'][i]
    # #print(best_seed_sets1)

    # max_influences1 = [0]*(max_budget+1)
    # for i,j in enumerate(budgets):
    #     max_influences1[j] = output_dict['genetic_exp_influences'][i]

    # output_dict['genetic_best_seed_sets'] = best_seed_sets1
    # output_dict['genetic_exp_influences'] = max_influences1
    # print(output_dict['genetic_exp_influences'])
    # print(len(output_dict['genetic_exp_influences']))
    # ""

    "saving output as a dataframe"
    if not os.path.exists(results_folder + os.sep + "plots"):
        os.makedirs(results_folder + os.sep + "plots")
    columns = list(output_dict.keys())
    output_df = pd.DataFrame()
    for column in columns:
        output_df[column] = output_dict[column]

    "subseting output_df for rows in [0,1] and then 5, 10, ..."
    indices = sorted(
        list(set([0, 1] + list(range(interval, max_budget + 1, interval))))
    )
    output_df = output_df.loc[indices, :]

    "saving output_df as a csv file"
    output_df.to_csv(
        results_folder
        + os.sep
        + "plots"
        + "/sim_exp_influence_data_"
        + str(max_budget)
        + ".csv",
        index=False,
        header=True,
    )

    "plotting"
    chosen = [colname for colname in output_df.columns if "exp_" in colname]
    ax = output_df.loc[:, chosen].plot(lw=2, marker="o")
    ax.set_xlabel("$k$")
    ax.set_ylabel("Expected Influence")
    legend_entries = [
        "CELF++",
        "Community-IM",
        "Genetic",
        "CoFIM",
        "Degree",
        "Degree-Discount",
    ]
    chosen_legend_entries = [
        item for i, item in enumerate(legend_entries) if i in chosen_algorithm_indices
    ]
    ax.legend(chosen_legend_entries)

    ax.grid(linestyle="-", linewidth=1)
    ax.set_title("Influence maximization for " + name_id[1:] + " network")
    # ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.get_figure().savefig(
        results_folder
        + os.sep
        + "plots"
        + "/sim"
        + name_id
        + "_plot_"
        + str(max_budget)
        + ".eps"
    )
    ax.get_figure().savefig(
        results_folder
        + os.sep
        + "plots"
        + "/sim"
        + name_id
        + "_plot_"
        + str(max_budget)
        + ".jpg"
    )
    # TODO fix these lines at some point
    # matplotlib.clean_figure()
    # matplotlib.save(
    #    results_folder
    #    + os.sep
    #    + "plots"
    #    + "/sim"
    #    + name_id
    #    + "_plot_"
    #    + str(max_budget)
    #    + ".tex"
    # )
