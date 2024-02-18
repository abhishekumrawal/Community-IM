#!/usr/bin/env python3

import random

import networkx as nx
import pytest

from im_functions.c_celfpp_nb import community_celf, fast_cascade


def test_community_celf() -> None:
    n = 1000
    p = 0.1
    k = 10
    test_graph = nx.fast_gnp_random_graph(n, p)

    community_celf(test_graph, k)
