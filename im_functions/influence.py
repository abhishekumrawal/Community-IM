import networkx as nx
import numpy as np

from cynetdiff.utils import networkx_to_ic_model

from im_functions.linear_threshold import linear_threshold


def influence(network, seed_set, diffusion_model, spontaneous_prob=[]):
    nodes = list(nx.nodes(network))
    spontaneously_infected = []
    influence = 0

    if len(spontaneous_prob) != 0:
        for m in range(len(network)):
            if np.random.rand() < spontaneous_prob[m]:
                spontaneously_infected.append(nodes[m])

    if diffusion_model == "independent_cascade":
        model = networkx_to_ic_model(network, activation_prob=0.2)
        model.set_seeds(list(set(spontaneously_infected + seed_set)))
        model.advance_until_completion()
        influence = model.get_num_activated_nodes()

    elif diffusion_model == "linear_threshold":
        layers = linear_threshold(network, list(set(spontaneously_infected + seed_set)))
        influence = np.sum([len(item) for item in layers])

    return influence
