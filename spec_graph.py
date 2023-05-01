import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


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


def spectral_drawing(fp, p, eps):
    A = np.loadtxt(fp)
    n = A.shape[0]
    # n = G.number_of_nodes()
    # A = nx.adjacency_matrix(G).toarray()
    d = np.sum(A, axis=1)
    D = np.diag(d)
    EDiA = 0.5 * (np.eye(n) + np.diag(1. / d) @ A)
    u = np.zeros(shape=(p, n))
    u[0] = np.full(n, 1. / np.sqrt(n))
    # for k in range(p - 1):
    #     u[k + 1] = np.random.default_rng().uniform(size=n)
    #     u[k + 1] = u[k + 1] / np.linalg.norm(u[k + 1])
    #     for l in range(k + 1):
    #         u[k + 1] = u[k + 1] - (u[k + 1] @ D @ u[l]) / (u[l] @ D @ u[l]) * u[l]
    #     u_old = u[k]
    #     while np.dot(u_old, u[k + 1]) < 1 - eps:
    #         u_old = u[k + 1].copy()
    #         u[k + 1] = EDiA @ u[k + 1]
    #         u[k + 1] = u[k + 1] / np.linalg.norm( u[k + 1])
    #     u[k + 1] /= np.sign(u[k + 1, 0])
    rng = np.random.default_rng()
    for k in range(p - 1):
        print(k)
        u_k = rng.uniform(size=n)
        u_k = u_k / np.linalg.norm(u_k)
        while True:
            u[k + 1] = u_k.copy()
            for l in range(k + 1):
                u[k + 1] = u[k + 1] - (u[k + 1] @ D @ u[l]) / (u[l] @ D @ u[l]) * u[l]
                x = u[k + 1] @ D @ u[l]
                y = np.dot(u[k + 1], D @ u[l])
            u_k = EDiA @ u[k + 1]
            u_k = u_k / np.linalg.norm(u_k)
            # print(u_k - u[k+1])
            if np.dot(u_k, u[k + 1]) >= 1 - eps:
                break
        u[k + 1] = u_k.copy()
        u[k + 1] /= np.sign(u[k + 1, 0])

    G = nx.Graph()
    for i, row in enumerate(A):
        for j, e in enumerate(row):
            if j > i and e == 1:
                G.add_edge(i, j)

    pos = nx.random_layout(G)
    for i in G.nodes:
        pos[i] = u[1:, i]

    t = (np.linalg.eig(np.diag(1. / d) @ A)[1])
    t = t[:, 2:4].T
    print(t)

    pos1 = nx.random_layout(G)
    for i in G.nodes:
        pos1[i] = t[:, i]

    return G, pos, u[1:, ], pos1


# path = 'graphs/graph254.txt'
# G = parse_adjacency_list(path)
# u = spectral_drawing(G, 3, 0.001)
G, pos, u, pos1 = spectral_drawing('graphs/graph1158_am.txt', 3, 1e-4)
print(u)
print(9)
# plt.scatter(u[0, :], u[1, :])
# plt.subplots()
fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, ax=ax)
fig1, ax1 = plt.subplots()
nx.draw_networkx(G, pos=nx.spectral_layout(G), ax=ax1)
# print(nx.adjacency_matrix(G).toarray())
print(nx.spectral_layout(G))
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
ax1.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
plt.show()