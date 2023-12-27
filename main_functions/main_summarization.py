#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:49:19 2019

@author: abhishek.umrawal
"""


def main_summarization(
    max_budget,
    diffusion_models,
    weighting_schemes,
    interval,
    all_algorithms,
    all_name_ids,
):
    "importing required built-in modules"
    import os as os
    import pickle

    import numpy as np
    import pandas as pd

    # TODO fix this function bc it's totally broken
    return

    "modified inputs"
    algorithms = all_algorithms[0:3]

    """
    summarizing max expected influences for all networks
    """

    "results folder"
    results_folder = (
        "results/results_" + diffusion_models[0] + "_" + weighting_schemes[0]
    )

    "looking up for nested non-adaptive methods"
    output_dict = {}
    output_dict["network"] = []
    for algorithm in algorithms:
        output_dict[algorithm] = []

    for name_id in all_name_ids:
        for algorithm in algorithms:
            filename = "runtime_info_" + algorithm + ".txt"
            filename_with_path = (
                results_folder
                + os.sep
                + "results"
                + name_id
                + os.sep
                + "runtime_files"
                + os.sep
                + filename
            )

            if os.path.exists(filename_with_path):
                with open(filename_with_path, "rb") as f:
                    results_dict = eval(f.read())

                if name_id[1:] not in output_dict["network"]:
                    output_dict["network"] += [name_id[1:]]

                output_dict[algorithm] += [int(round(results_dict[algorithm], 0))]

    "saving output as a dataframe"
    columns = list(output_dict.keys())
    output_df = pd.DataFrame()

    for column in columns:
        output_df[column] = output_dict[column]
    output_df["ratio_with_celfpp"] = round(
        output_df["celfpp"] / output_df["ccelfpp1"], 2
    )
    output_df["ratio_with_cofim"] = round(output_df["cofim"] / output_df["ccelfpp1"], 2)
    output_df.loc["average"] = round(output_df.mean(), 2)

    "saving output_df as a csv file"
    output_df.to_csv(
        results_folder + os.sep + "summary_runtimes.csv",
        index=False,
        header=True,
        sep=",",
    )
    print(output_df.to_latex(index=False))

    """
    summarizing max expected influences for all networks
    """

    "chosen budgets"
    indices = list(set([0, 1] + list(range(interval, max_budget + 1, interval))))

    "looking up for nested non-adaptive methods"
    output_dict = {}
    output_dict["network"] = []
    for algorithm in algorithms:
        output_dict[algorithm] = []

    for name_id in all_name_ids:
        for algorithm in algorithms:
            filename = "output_" + algorithm + "__" + str(max_budget) + "__.pkl"
            filename_with_path = (
                results_folder
                + os.sep
                + "results"
                + name_id
                + os.sep
                + "pickle_files"
                + os.sep
                + filename
            )

            if os.path.exists(filename_with_path):
                with open(filename_with_path, "rb") as f:
                    results_dict = pickle.load(f)

                if name_id[1:] not in output_dict["network"]:
                    output_dict["network"] += [name_id[1:]]

                output_dict[algorithm] += [
                    round(
                        np.max(
                            [
                                item
                                for i, item in enumerate(results_dict["exp_influence"])
                                if i in indices
                            ]
                        ),
                        2,
                    )
                ]

    "saving output as a dataframe"
    columns = list(output_dict.keys())
    output_df = pd.DataFrame()
    for column in columns:
        output_df[column] = output_dict[column]
    output_df = output_df[(output_df.T != 0).all()]
    output_df["ratio_with_celfpp"] = round(
        output_df["ccelfpp1"] / output_df["celfpp"], 2
    )
    output_df["ratio_with_cofim"] = round(output_df["ccelfpp1"] / output_df["cofim"], 2)
    output_df.loc["average"] = round(output_df.mean(), 2)

    "saving output_df as a csv file"
    output_df.to_csv(
        results_folder + os.sep + "summary_max_exp_influences.csv",
        index=False,
        header=True,
        sep=",",
    )
    print(output_df.to_latex(index=False))
