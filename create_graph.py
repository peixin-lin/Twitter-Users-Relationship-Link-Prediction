import networkx as nx
import numpy as np
import timeit

time0 = timeit.default_timer()
# Read the pairs
with np.load('filtered_data.npz') as fd:
    pairs = fd['pairs']

time1 = timeit.default_timer()
print('Time for reading file: ', time1 - time0)

# Create a digraph for the task
DG = nx.DiGraph()
DG.add_edges_from(pairs)

# Create a undirected graph for computing AA, JC and RA
UDG = nx.Graph()
UDG.add_edges_from(pairs)

time2 = timeit.default_timer()
print('Time for creating graphs: ', time2 - time1)

print("Num. nodes: ", nx.number_of_nodes(DG))
print("Num. edges: ", nx.number_of_edges(DG))

# Get nodes edges, and non-edges
nodes = nx.nodes(DG)
edges = nx.edges(DG)
non_edges = nx.non_edges(DG)
edges = nx.edges(UDG)

# Compute HAA, HJC and HRA
count = 0
HAA = []
HJC = []
HRA = []
for e in edges:
    try:
        AA = nx.adamic_adar_index(UDG, [e])
    except ZeroDivisionError:
        AA = 0

    try:
        JC = nx.jaccard_coefficient(UDG, [e])
    except ZeroDivisionError:
        JC = 0

    try:
        RA = nx.resource_allocation_index(UDG, [e])
    except ZeroDivisionError:
        RA = 0
    SD = DG.in_degree(e[1]) - DG.in_degree(e[0])  # specificity_difference
    if SD < 0:
        HAA.append(0)
        HJC.append(0)
        HRA.append(0)
    else:
        for u, v, p in AA:
            HAA.append(p)
        for u, v, p in JC:
            HJC.append(p)
        for u, v, p in RA:
            HRA.append(p)

time3 = timeit.default_timer()

print('Time for calculating features: ', time3 - time2)

# Store the feature scores
np.savez_compressed("features_positive", HAA=HAA, HJC=HJC, HRA=HRA)
