#!/usr/bin/env python3

import random

import networkx as nx
import pytest

from im_functions.independent_cascade import independent_cascade


@pytest.mark.parametrize("use_fast_impl", [True, False])
def test_independent_cascade(benchmark, use_fast_impl: bool) -> None:
    n = 10
    p = 0.1
    k = 10
    test_graph = nx.fast_gnp_random_graph(n, p)

    for u, v, data in test_graph.edges(data=True):
        act_prob = data.setdefault("act_prob", 0.1)
        data["success_prob"] = random.random()
        if act_prob > 1.0:
            raise Exception()

    nodes = list(test_graph.nodes)
    seeds = random.sample(nodes, k)

    activated_nodes_levels = independent_cascade(
        test_graph, seeds, graphblas_impl=False
    )

    cascade_res = benchmark(
        independent_cascade, test_graph, seeds, graphblas_impl=use_fast_impl
    )
    for level_a, level_b in zip(activated_nodes_levels, cascade_res):
        assert set(level_a) == set(level_b)
