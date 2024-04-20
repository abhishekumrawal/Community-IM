import heapq as hq
import random

import networkx as nx
import numba as nb
import numba.typed as nbt
import numpy as np


def convert_to_csr(graph: nx.Graph | nx.DiGraph) -> tuple[np.ndarray, np.ndarray]:
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}

    starts = []
    edges = []
    curr_start = 0

    for node in graph.nodes():
        starts.append(curr_start)
        for neighbor in graph.neighbors(node):
            curr_start += 1
            edges.append(node_mapping[neighbor])

    return np.array(starts), np.array(edges)


def community_celf(G: nx.Graph | nx.DiGraph, budget: int) -> set[int]:

    if not G.is_directed():
        G = G.to_directed()

    starts, edges = convert_to_csr(G)
    community = set(G.nodes())

    return _community_celf(starts, edges, community, budget)


def _community_celf(
    starts: np.ndarray, edges: np.ndarray, community: set[int], budget: int
) -> set[int]:
    seeds = {0}

    curr_best = -1
    last_seed = -1
    curr_best_val = 0

    node_heap: list[tuple[int, int, int, int, int]] = [(0, 0, curr_best, 0, 0)]
    hq.heappop(node_heap)

    for v in range(len(starts)):
        new_seeds = {v}
        if curr_best != -1:
            new_seeds.add(curr_best)

        mg1 = fast_cascade(starts, edges, new_seeds, [0])
        curr_tup = (
            -mg1,
            0,
            -fast_cascade(starts, edges, new_seeds, [0]),
            curr_best,
            0,
        )
        hq.heappush(node_heap, curr_tup)

        if mg1 > curr_best_val:
            curr_best_val = mg1
            curr_best = v

    while len(seeds) < budget:
        mg1, mg2, prev_best, flag, u = hq.heappop(node_heap)
        mg1 = -mg1
        mg2 = -mg2

        if flag == len(seeds):
            last_seed = u
            seeds.add(u)
            continue
        elif prev_best == last_seed:
            mg1 = mg2
        else:
            seed_val = fast_cascade(starts, edges, seeds, [u])
            mg1 = fast_cascade(starts, edges, seeds, [u]) - seed_val
            mg2 = fast_cascade(starts, edges, seeds, [u])

        flag = len(seeds)
        curr_tup = (
            -mg1,
            -mg2,
            curr_best,
            flag,
            u,
        )
        hq.heappush(node_heap, curr_tup)

        if mg1 > curr_best_val:
            curr_best_val = mg1
            curr_best = u

    return seeds


@nb.njit
def fast_cascade(
    starts: np.ndarray, edges: np.ndarray, seeds: set[int], new_nodes: list[int]
) -> int:

    visited = set(seeds)
    current_layer = nbt.List(seeds)

    while current_layer:
        next_layer = nbt.List.empty_list(nb.int64)

        for node in current_layer:

            range_end = edges.shape[0]
            if node + 1 < starts.shape[0]:
                range_end = starts[node + 1]

            for i in range(starts[node], range_end):
                child = edges[i]
                print(child)
                if child not in visited:
                    if np.random.random() <= 0.1:
                        visited.add(child)
                        next_layer.append(child)
        current_layer = next_layer

    return len(visited)
