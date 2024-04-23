from collections import defaultdict
import operator
import copy


def degdiscount_im(network, budget):
    budget = min(budget, len(network.nodes))
    S = set()
    d = dict(network.out_degree)
    dd = copy.deepcopy(d)
    t = defaultdict(int)
    for _ in range(budget):
        u = max(dd.items(), key=operator.itemgetter(1))[0]
        S = S.union({u})
        del dd[u]
        for v in network.neighbors(u):
            if v not in S:
                t[v] += 1
                dd[v] = (
                    d[v]
                    - 2 * t[v]
                    - (d[v] - t[v]) * t[v] * network.edges[u, v]["act_prob"]
                )
    return list(S)
