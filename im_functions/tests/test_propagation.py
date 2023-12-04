#!/usr/bin/env python3

import networkx as nx
from im_functions.propagation.independent_cascade import independent_cascade
from im_functions.propagation.independent_cascade_graphblas import independent_cascade_fast
import random

def test_independent_cascade() -> None:
    n = 100
    p = 0.2
    test_graph = nx.fast_gnp_random_graph(n, p)
    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, 10)
    activated_nodes_levels = independent_cascade(test_graph, seeds, random_seed=1234)

    for _ in range(10):
        assert activated_nodes_levels == independent_cascade_fast(test_graph, seeds, random_seed=1234)
