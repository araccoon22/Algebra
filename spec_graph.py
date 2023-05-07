import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import scipy as sp


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


def parse_adjacency_matrix(filepath, delims=','):
    G = nx.Graph()
    f = open(filepath, 'r', )
    for i, s in enumerate(f):
        weights = s.split(delims[0])
        for j, weight in enumerate(weights):
            if i == j:
                G.add_node(i)
            if j > i:
                if float(weight) != 0:
                    G.add_edge(i, j)

    return G


def spectral_drawing(G, p, eps=1e-8, maxiter=5000, seed=None):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    d = np.array([G.degree(node) for node in G.nodes()])
    B = 0.5 * (sp.sparse.csr_array(np.eye(n)) + sp.sparse.diags(1. / d) @ A)
    u = np.zeros(shape=(p, n))
    u[0] = np.full(n, 1. / np.sqrt(n))
    rng = np.random.default_rng(seed=seed)
    for k in range(p - 1):
        # Generate random starting vector
        u[k + 1] = rng.normal(size=n)
        u[k + 1] /= np.linalg.norm(u[k + 1])
        i = 0
        while i < maxiter:
            i += 1
            # D-orthogonalization with respect to previous eigenvectors
            u[k + 1] = u[k + 1] - (u[k + 1] * d @ u[0:k+1].T) / np.einsum('ij,ij->i', u[0:k+1] * d, u[0:k+1]) @ u[0:k+1]
            u_old = u[k + 1].copy()
            # Power iteration
            u[k + 1] = B @ u[k + 1]
            # Normalization
            u[k + 1] /= np.linalg.norm(u[k + 1])
            if np.dot(u_old, u[k + 1]) >= 1 - eps:
                break
        # Making sure that the signs are always the same
        u[k + 1] /= np.sign(u[k + 1, 0])

    # Creating layout for G
    pos = nx.random_layout(G)
    for i, node in enumerate(G.nodes):
        pos[node] = u[1:, i]

    return pos, u[1:, ]


graph_id = 8
path = 'graphs/graph' + str(graph_id) + '.txt'
# G = parse_adjacency_list(path)
G = parse_adjacency_matrix(path)
pos, u = spectral_drawing(G, 3)
# G, pos, u, pos1 = spectral_drawing('graphs/graph19217_am.txt', 3, 1e-8)
print(np.round(u, 2))
fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, ax=ax, node_size=50, linewidths=0.5)
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

pos1, u1 = spectral_drawing(G, 4)
node_xyz = np.array([pos1[v] for v in sorted(G)])
edge_xyz = np.array([(pos1[u], pos1[v]) for u, v in G.edges()])
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection="3d")
ax1.scatter(*node_xyz.T)
for vizedge in edge_xyz:
    ax1.plot(*vizedge.T, color="tab:gray")
plt.show()