import heapq as hq
import random

import networkx as nx
import numba as nb


@nb.njit
def community_celf(
    G: nx.Graph | nx.DiGraph, communities: list[list[int]], budget: int
) -> list[int]:
    seeds = set()
    last_seed = None
    curr_best = None

    # Tuple in heap is (mg1, mg2, prev_best, flag)
    node_heap = []
    for v in G.nodes():
        new_seeds = [v]
        if curr_best is not None:
            new_seeds.append(curr_best)

        curr_tup = (
            -_fast_cascade(G, [v]),
            -_fast_cascade(G, new_seeds),
            curr_best,
            0,
            v,
        )
        hq.heappush(node_heap, curr_tup)
        curr_best = node_heap[0][-1]

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
            seed_val = _fast_cascade(G, list(seeds))
            mg1 = _fast_cascade(G, list(seeds) + [u]) - seed_val
            mg2 = _fast_cascade(G, list(seeds) + [u, curr_best])

        flag = len(seeds)
        curr_tup = (
            -mg1,
            -mg2,
            curr_best,
            flag,
            u,
        )
        hq.heappush(node_heap, curr_tup)
        curr_best = node_heap[0][-1]

    return seeds


def _fast_cascade(G: nx.Graph | nx.DiGraph, seeds: list[int]) -> list[list[int]]:
    if not G.is_directed():
        G = G.to_directed()

    visited = set(seeds)

    # this is basically BFS, except that the current layer only stores the nodes at
    # same distance from sources at each iteration
    res = []
    current_layer = seeds.copy()

    while current_layer:
        res.append(current_layer)
        current_layer = []

        for next_node in res[-1]:
            for child, data in G[next_node].items():
                if child not in visited:
                    # Lazy getter to deal with not having this set but still being
                    # efficient
                    succ_prob = data.get("success_prob")
                    if succ_prob is None:
                        succ_prob = random.random()

                    if succ_prob <= data.get("act_prob", 0.1):
                        visited.add(child)
                        current_layer.append(child)

    return res
