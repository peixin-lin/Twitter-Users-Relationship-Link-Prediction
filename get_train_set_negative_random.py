import networkx as nx
import numpy as np
import timeit

time0 = timeit.default_timer()
'''Read the pairs'''
with np.load('filtered_data.npz') as fd:
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

'''Get nodes edges, and non-edges'''
non_edges = nx.non_edges(UDG)
sample_edges = []
for n in non_edges:
    sample_edges.append(n)
'''Compute HAA, HJC and HRA'''
HAA = []
HJC = []
HRA = []
count = 0

sample_edges = np.random.choice(a=sample_edges, size=5000000, replace=False)
for e in sample_edges:
    if count == 5000000:
        break
    if count % 1000 == 0:
        print(count)
    count += 1
    AA = nx.adamic_adar_index(UDG, [e])
    JC = nx.jaccard_coefficient(UDG, [e])
    RA = nx.resource_allocation_index(UDG, [e])
    SD = DG.in_degree(e[1]) - DG.in_degree(e[0])  # specificity_difference
    if SD < 0:
        HAA.append(0)
        HJC.append(0)
        HRA.append(0)
    else:
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
np.savez_compressed("train_features_negative_random", HAA=HAA, HJC=HJC, HRA=HRA)