import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from random import randint
from spec_graph import *


def gen_stoch_block_graph(n, r, sizes=None, prob=None, seed=None):
    rng = np.random.default_rng(seed=seed)

    if prob is None:
        # prob = rng.uniform(size=(r, r)) / 5
        # prob = (prob + prob.T) / 2
        # prob[np.diag_indices_from(prob)] = 0.3 + rng.uniform(size=r) / 2

        prob = rng.uniform(size=(r, r)) / 2
        prob = (prob + prob.T) / 2
        prob[np.diag_indices_from(prob)] = rng.uniform(size=r)
    assert prob.shape == (r, r)

    if sizes is None:
        group_ind = np.sort(rng.choice(np.linspace(1, n - 2, n - 2, dtype=int), size=r-1, replace=False))
        group_ind = np.hstack(([0], group_ind, [n]))
    else:
        assert np.sum(sizes) == n
        group_ind = np.hstack(([0], np.cumsum(sizes)))

    G = nx.Graph()
    for j in range(r):
        for i in range(group_ind[j], group_ind[j + 1]):
            G.add_node(i, block=j)

    check = np.zeros((r, r))
    for i in range(n):
        for j in range(i + 1, n):
            edge_prob = prob[G.nodes[i]['block'], G.nodes[j]['block']]
            if rng.random() < edge_prob:
                G.add_edge(i, j)
                check[G.nodes[i]['block'], G.nodes[j]['block']] += 1

    return G, prob, check


sizes = [75, 75, 300]
probs = np.array([[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]])
# G = nx.stochastic_block_model(sizes, probs, seed=0)
# G, p, c = gen_stoch_block_graph(450, 3, sizes=np.array(sizes), prob=probs, seed=0)

r = 5
G, p, c = gen_stoch_block_graph(1000, r)
u = spectral_drawing(G, 4)
groups = [[] for _ in range(r)]
for node in G.nodes().items():
    groups[node[1]['block']].append(u[:, node[0]])

fig, ax = plt.subplots()
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection='3d')
for i in range(r):
    g = np.array(groups[i])
    ax.scatter(g[:, 0], g[:, 1], label='Block' + str(i))
    ax1.scatter(g[:, 0], g[:, 1], g[:, 2], label='Block'+str(i))
ax1.legend()
ax.legend()

plot_spectral_drawing(G)
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=r, random_state=0)
# labels = kmeans.fit_predict(u.T)
# u_labels = np.unique(labels)
#
# fig2, ax2 = plt.subplots()
# for i in u_labels:
#     ax2.scatter(u[:, labels == i][0], u[:, labels == i][1], label=i)
# ax2.legend()

plt.show()

print(1)