import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy.sparse import *


def parse_adjacency_list(filepath, delims=': '):
    G = nx.Graph()
    f = open(filepath, 'r', )
    for s in f:
        verts = s.split(delims[0])
        v = int(verts[0])
        G.add_node(v)
        adj = verts[1].split(delims[1])
        for w in adj:
            G.add_edge(v, int(w))

    return G


def spectral_drawing(G, p, eps=1e-8, maxiter=5000, seed=None):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    d = np.array([G.degree(node) for node in G.nodes()])
    B = 0.5 * (csr_array(np.eye(n)) + diags(1. / d) @ A)
    u = np.zeros(shape=(p, n))
    u[0] = np.full(n, 1. / np.sqrt(n))
    rng = np.random.default_rng(seed=seed)
    for k in range(p - 1):
        u[k + 1] = rng.normal(size=n)
        u[k + 1] /= np.linalg.norm(u[k + 1])
        i = 0
        while i < maxiter:
            i += 1
            u[k + 1] = u[k + 1] - (u[k + 1] * d @ u[0:k+1].T) / np.einsum('ij,ij->i', u[0:k+1] * d, u[0:k+1]) @ u[0:k+1]
            u_old = u[k + 1].copy()
            u[k + 1] = B @ u[k + 1]
            u[k + 1] /= np.linalg.norm(u[k + 1])
            if np.dot(u_old, u[k + 1]) >= 1 - eps:
                break
        u[k + 1] /= np.sign(u[k + 1, 0])

    pos = nx.random_layout(G)
    for i, node in enumerate(G.nodes):
        pos[node] = u[1:, i]

    return pos, u[1:, ]


graph_id = 3345
path = 'graphs/graph' + str(graph_id) + '.txt'
G = parse_adjacency_list(path)
pos, u = spectral_drawing(G, 3)
# G, pos, u, pos1 = spectral_drawing('graphs/graph19217_am.txt', 3, 1e-8)
print(np.round(u, 2))
fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, ax=ax)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()