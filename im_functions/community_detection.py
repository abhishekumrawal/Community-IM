#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:37:20 2019

@author: cjquinn
"""

import copy
import os
import random

import igraph as ig
import leidenalg
import louvain
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import (
    girvan_newman,
    greedy_modularity_communities,
    label_propagation_communities,
)

# https://networkx.github.io/documentation/stable/reference/algorithms/community.html


def community_detection(network, method="louvain"):
    random.seed(0)

    # flow-based community detection
    if method == "infomap":
        # writing the given network as a .net file
        nx.write_pajek(network, "infomap_input/network.net")

        # running Infomap through termminal to detect the communities
        os.system("infomap -d infomap_input/network.net infomap_output --clu")

        # reading the results of Infomap
        data = pd.read_table("infomap_output/network.clu", skiprows=range(1))

        # extracting the nodes and community labels
        nodes = []
        community_labels = []
        for row in data.iloc[7:, 0]:
            nodes.append(int(row.split(" ")[0]))
            community_labels.append(int(row.split(" ")[1]))
        nodes = [x - 1 for x in nodes]

        # creating a list of communities: a community is a list of nodes in that community
        communities = []
        for label in list(np.unique(np.array(community_labels))):
            communities.append(
                [nodes[i] for i, x in enumerate(community_labels) if x == label]
            )

    # modularity based community detection
    if method == "louvain":
        # copy the given network from newtorkx object to an igraph object
        # all edge attributes like direction and weight are preserved
        # use G.vs for vertices and G.es for edges
        G = ig.Graph.from_networkx(network)

        part = louvain.find_partition(
            G, louvain.ModularityVertexPartition, weights="act_prob"
        )
        communities_igraph_lables = list(part)

        # setting communities node lables same as that in the networkx object
        communities = copy.deepcopy(communities_igraph_lables)
        for i, com in enumerate(communities_igraph_lables):
            for j, num in enumerate(com):
                communities[i][j] = G.vs(num)["_nx_name"][0]

    if method == "leiden":
        # copy the given network from newtorkx object to an igraph object
        # all edge attributes like direction and weight are preserved
        # use G.vs for vertices and G.es for edges
        G = ig.Graph.from_networkx(network)

        part = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition, weights="act_prob"
        )
        communities_igraph_lables = list(part)

        # setting communities node lables same as that in the networkx object
        communities = copy.deepcopy(communities_igraph_lables)
        for i, com in enumerate(communities_igraph_lables):
            for j, num in enumerate(com):
                communities[i][j] = G.vs(num)["_nx_name"][0]

    if method == "greedy_modularity":
        part = greedy_modularity_communities(network, weight="act_prob")
        communities = [list(item) for item in part]

    # semi-supervised learning-based community detection
    if method == "label_propagation":
        part = label_propagation_communities(nx.Graph(network))
        communities = [list(item) for item in part]

    if method == "girvan_newman":
        part = girvan_newman(nx.Graph(network))
        communities = list(sorted(item) for item in next(part))

    return communities
