from spec_graph import *


graph_id = 1041
path = f'graphs/graph{graph_id}.txt'
G = parse_adjacency_list(path)
# G = parse_adjacency_matrix(path)
# plot_spectral_drawing(G, plot3d=True)

plot_spectral_drawing(G, plot3d=True)
# pos, u = spectral_drawing(G, 3)
# G, pos, u, pos1 = spectral_drawing('graphs/graph19217_am.txt', 3, 1e-8)
# print(np.round(u, 2))
# fig, ax = plt.subplots()
# nx.draw_networkx(G, pos=pos, ax=ax, node_size=50, linewidths=0.5)
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
# fig2, ax2 = plt.subplots()
# nx.draw_networkx(G, pos=nx.spectral_layout(G), ax=ax2, node_size=50, linewidths=0.5)
#
# pos1, u1 = spectral_drawing(G, 4)
# node_xyz = np.array([pos1[v] for v in sorted(G)])
# edge_xyz = np.array([(pos1[u], pos1[v]) for u, v in G.edges()])
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection="3d")
# ax1.scatter(*node_xyz.T)
# for vizedge in edge_xyz:
#     ax1.plot(*vizedge.T, color="tab:gray", linewidth=0.5)
plt.show()

