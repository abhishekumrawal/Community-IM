#!/usr/bin/env python3

import random
import time

import networkx as nx

from im_functions.independent_cascade import independent_cascade


def test_independent_cascade() -> None:
    n = 1000
    p = 0.1
    k = 10
    test_graph = nx.fast_gnp_random_graph(n, p)

    for u, v, data in test_graph.edges(data=True):
        act_prob = data.setdefault("act_prob", 0.1)
        data["success_prob"] = random.random()
        if act_prob > 1.0:
            raise Exception()

    # TODO add benchmarks to these tests
    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)
    print("Starting")
    start = time.perf_counter()
    activated_nodes_levels = independent_cascade(
        test_graph, seeds, graphblas_impl=False
    )
    end = time.perf_counter()
    print("Slow time:", end - start)

    start = time.perf_counter()
    independent_cascade(test_graph, seeds)
    end = time.perf_counter()
    print("Fast time:", end - start)
    # assert False
    for _ in range(10):
        for level_a, level_b in zip(
            activated_nodes_levels, independent_cascade(test_graph, seeds)
        ):
            assert set(level_a) == set(level_b)

    # assert False
