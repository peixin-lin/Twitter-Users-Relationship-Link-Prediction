import networkx as nx
import numpy as np
import timeit
from priority_queue import PriorityQueue as pq

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
candidates = pq()
count = 0
selected = 0
for ne in non_edges:
    AA = nx.adamic_adar_index(UDG, [ne])
    count += 1
    if selected == 800000:
        break
    if count % 6 == 0:
        try:
            for u, v, p in AA:
                candidates.push((u, v), -p)
                selected += 1
                if count % 10000 == 0:
                    print('Unsorted instances selected: ', selected, 'out of ', count)

        except ZeroDivisionError:
            candidates.push((u, v), 0)
            pass


'''Compute HAA, HJC and HRA'''
HAA = []
HJC = []
HRA = []
SD = []
for i in range(100000):
    if i % 10000 == 0:
        print('Sorted instances selected: ', i)

    e = candidates.pop()
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
np.savez_compressed("new_negative", HAA=HAA, HJC=HJC, HRA=HRA, SD=SD)