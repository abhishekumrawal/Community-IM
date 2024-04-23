import copy
import random

import networkx as nx

__all__ = ["independent_cascade"]


def independent_cascade(
    G, seeds, *, steps=0, random_seed=None, fast_impl: bool = True
) -> list[list[int]]:
    """Return the active nodes of each diffusion step by the independent cascade model

    Parameters
    -----------
    G : graph
        A NetworkX graph
    seeds : list of nodes
        The seed nodes for diffusion
    steps: integer
        The number of steps to diffuse.  If steps <= 0, the diffusion runs until
        no more nodes can be activated.  If steps > 0, the diffusion runs for at
        most "steps" rounds

    Returns
    -------
    layer_i_nodes : list of list of activated nodes
        layer_i_nodes[0]: the seeds
        layer_i_nodes[k]: the nodes activated at the kth diffusion step

    Notes
    -----
    When node v in G becomes active, it has a *single* chance of activating
    each currently inactive neighbor w with probability p_{vw}

    Examples
    --------
    DG = nx.DiGraph()
    DG.add_edges_from([(1,2), (1,3), (1,5), (2,1), (3,2), (4,2), (4,3), (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], act_prob=0.2)
    layers = networkx_addon.information_propagation.independent_cascade(DG, [6])

    References
    ----------
    [1] David Kempe, Jon Kleinberg, and Eva Tardos.
        Influential nodes in a diffusion model for social networks.
        In Automata, Languages and Programming, 2005.
    """

    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise Exception(
            "independent_cascade() is not defined for graphs with multiedges."
        )

    if fast_impl:
        return _fast_cascade(G, seeds)

    rand_gen = random.Random(random_seed)

    # change to directed graph
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # init activation probabilities
    for u, v, data in DG.edges(data=True):
        act_prob = data.setdefault("act_prob", 0.1)
        # if "act_prob" not in data:
        #    data["act_prob"] = 0.1
        if act_prob > 1.0:
            raise Exception(
                f"edge activation probability: {act_prob} cannot be larger than 1."
            )

        data.setdefault("success_prob", rand_gen.random())

    # perform diffusion
    A = copy.deepcopy(seeds)  # prevent side effect
    if steps <= 0:
        # perform diffusion until no more nodes can be activated
        return _diffuse_all(DG, A, rand_gen)
    # perform diffusion for at most "steps" rounds
    return _diffuse_k_rounds(DG, A, steps, rand_gen)


def _diffuse_all(G, A, rand_gen):
    tried_edges = set()
    layer_i_nodes = []
    layer_i_nodes.append([i for i in A])  # prevent side effect
    while True:
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(
            G, A, tried_edges, rand_gen
        )
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            break
    return layer_i_nodes


def _diffuse_k_rounds(G, A, steps, rand_gen):
    tried_edges = set()
    layer_i_nodes = []
    layer_i_nodes.append([i for i in A])
    while steps > 0 and len(A) < len(G):
        len_old = len(A)
        (A, activated_nodes_of_this_round, cur_tried_edges) = _diffuse_one_round(
            G, A, tried_edges, rand_gen
        )
        layer_i_nodes.append(activated_nodes_of_this_round)
        tried_edges = tried_edges.union(cur_tried_edges)
        if len(A) == len_old:
            break
        steps -= 1
    return layer_i_nodes


def _diffuse_one_round(G, A, tried_edges, rand_gen):
    activated_nodes_of_this_round = set()
    cur_tried_edges = set()
    for s in A:
        for nb in G.successors(s):
            if nb in A or (s, nb) in tried_edges or (s, nb) in cur_tried_edges:
                continue
            if _prop_success(G, s, nb, rand_gen):
                activated_nodes_of_this_round.add(nb)
            cur_tried_edges.add((s, nb))
    activated_nodes_of_this_round = list(activated_nodes_of_this_round)
    A.extend(activated_nodes_of_this_round)
    return A, activated_nodes_of_this_round, cur_tried_edges


def _prop_success(G, src, dest, rand_gen):
    return G[src][dest]["success_prob"] <= G[src][dest]["act_prob"]


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
