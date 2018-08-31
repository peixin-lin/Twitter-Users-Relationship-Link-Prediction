import networkx as nx
import pandas as pd
import numpy as np

with np.load('filtered_data.npz') as fd:
    pairs = fd['pairs']
g = nx.DiGraph()
g.add_edges_from(pairs)
print("Num. nodes: ", nx.number_of_nodes(g))
print("Num. edges: ", nx.number_of_edges(g))

nodes = nx.nodes(g)
edges = nx.edges(g)
non_edges = nx.non_edges(g)

for n in nodes:
    print(n)

# for e in edges:
#     print(e)

# for ne in non_edges:
#     print(ne)
