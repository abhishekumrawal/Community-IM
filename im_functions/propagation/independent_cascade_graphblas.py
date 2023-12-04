"""
Implement independent cascade model
"""
#!/usr/bin/env python
#    Copyright (C) 2004-2010 by
#    Hung-Hsuan Chen <hhchen@psu.edu>
#    All rights reserved.
#    BSD license.
#    NetworkX:http://networkx.lanl.gov/.
__author__ = """Hung-Hsuan Chen (hhchen@psu.edu)"""

import copy

import graphblas_algorithms as ga
import networkx as nx

__all__ = ["independent_cascade_fast"]


def independent_cascade_fast(G, seeds, *, steps=0, random_seed=None):
    """Return the active nodes of each diffusion step by the independent cascade
  model

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
  >>> DG = nx.DiGraph()
  >>> DG.add_edges_from([(1,2), (1,3), (1,5), (2,1), (3,2), (4,2), (4,3), \
  >>>   (4,6), (5,3), (5,4), (5,6), (6,4), (6,5)], act_prob=0.2)
  >>> layers = networkx_addon.information_propagation.independent_cascade(DG, [6])

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

    # make sure the seeds are in the graph
    for s in seeds:
        if s not in G.nodes():
            raise Exception("seed", s, "is not in graph")

    # change to directed graph
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # adjusted_list = {}
    edge_to_remove = []

    for u, v, data in DG.edges(data=True):
        # adj_list = adjusted_list.setdefault(u, [])
        if data["success_prob"] > data["act_prob"]:
            edge_to_remove.append((u, v))

    DG.remove_edges_from(edge_to_remove)

    DG2 = ga.Graph.from_networkx(DG)
    layers = ga.bfs_layers(DG2, seeds)
    return list(map(list, layers))
