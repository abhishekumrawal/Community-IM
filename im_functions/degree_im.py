def degree_im(network, budget):
    budget = min(budget, len(network.nodes))
    return [
        x[0]
        for x in sorted(
            dict(network.out_degree).items(), key=lambda item: item[1], reverse=True
        )
    ][:budget]
