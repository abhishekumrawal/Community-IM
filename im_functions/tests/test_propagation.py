#!/usr/bin/env python3

import random

import networkx as nx

from im_functions.propagation.independent_cascade import independent_cascade
from im_functions.propagation.independent_cascade_graphblas import (
    independent_cascade_fast,
)


def test_independent_cascade() -> None:
    n = 100
    p = 0.2
    test_graph = nx.fast_gnp_random_graph(n, p)

    for u, v, data in test_graph.edges(data=True):
        act_prob = data.setdefault("act_prob", 0.1)
        data["success_prob"] = random.random()
        if act_prob > 1.0:
            raise Exception()

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, 10)
    activated_nodes_levels = independent_cascade(test_graph, seeds, random_seed=1234)
    print(activated_nodes_levels)

    for _ in range(10):
        for level_a, level_b in zip(
            activated_nodes_levels, independent_cascade_fast(test_graph, seeds)
        ):
            assert set(level_a) == set(level_b)
