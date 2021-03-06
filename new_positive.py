import networkx as nx
import numpy as np
import timeit


time0 = timeit.default_timer()
'''Read the pairs'''
with np.load('original_pairs.npz') as fd:
    pairs = fd['pairs']

time1 = timeit.default_timer()
print('Time for reading file: ', time1 - time0)

'''Create a digraph for the task'''
DG = nx.DiGraph()
DG.add_edges_from(pairs)
'''Create a undirected graph for computing AA, JC and RA'''
UDG = nx.Graph()
UDG.add_edges_from(pairs)

time2 = timeit.default_timer()
print('Time for creating graphs: ', time2 - time1)

print("Num. nodes: ", nx.number_of_nodes(DG))
print("Num. edges: ", nx.number_of_edges(DG))

'''Get nodes edges, and non-edges'''
nodes = nx.nodes(DG)
edges = nx.edges(UDG)
non_edges = nx.non_edges(DG)
'''Compute HAA, HJC and HRA'''
HAA = []
HJC = []
HRA = []
SD = []
count = 0
for e in edges:
    if count == 30000:
        break
    if count % 1000 == 0:
        print(count)
    count += 1
    AA = nx.adamic_adar_index(UDG, [e])
    JC = nx.jaccard_coefficient(UDG, [e])
    RA = nx.resource_allocation_index(UDG, [e])
    spec_diff = DG.in_degree(e[1]) - DG.in_degree(e[0])  # specificity_difference
    SD.append(spec_diff)
    try:
        for u, v, p in AA:
            HAA.append(p)
    except ZeroDivisionError:
            HAA.append(0)
            pass

    try:
        for u, v, p in JC:
            HJC.append(p)
    except ZeroDivisionError:
            HJC.append(0)
            pass

    try:
        for u, v, p in RA:
            HRA.append(p)
    except ZeroDivisionError:
            HRA.append(0)
            pass


time3 = timeit.default_timer()

print('Time for calculating features: ', time3 - time2)

'''Store the feature scores'''
np.savez_compressed("new_positive_original", HAA=HAA, HJC=HJC, HRA=HRA, SD=SD)


