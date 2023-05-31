import networkx as nx
import numpy as np
from spec_graph import *


def gen_ba_graph(G0, m, k, seed=None):
    rng = np.random.default_rng(seed)
    m0 = G0.number_of_nodes()
    assert m <= m0

    deg = np.array([G0.degree(node) for node in G0.nodes()])
    node_prob = deg / (2 * G0.number_of_edges())
    for new_node in range(m0, m0 + k):
        deg_change = np.zeros(m, dtype=int)
        old_nodes = np.copy(G0.nodes())
        for j in range(m):
            successful_edge = False
            while not successful_edge:
                connection_node = rng.choice(old_nodes, p=node_prob)
                if not (new_node, connection_node) in G0.edges():
                    successful_edge = True
                    G0.add_edge(new_node, connection_node)
                    deg_change[j] = connection_node
        deg[deg_change] += 1
        deg = np.append(deg, m)
        node_prob = deg / (2 * G0.number_of_edges())

    print("done")
    return G0


m0 = 10
# G0 = nx.full_rary_tree(2, m0)
G0 = nx.complete_graph(m0)
# G0 = nx.circular_ladder_graph(m0)
# G = nx.barabasi_albert_graph(10000, 5, seed=0)
G = gen_ba_graph(G0, 2, 1000, seed=0)
d = np.array([G0.degree(node) for node in G0.nodes()])
plot_spectral_drawing(G, scale_to_deg=True)
print(1)