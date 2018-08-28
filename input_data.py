import numpy as np
import pandas as pd
import os
# print(os.listdir("input"))

vertex_set = set();
sink_dict = {};

with open(r"../data/train.txt") as train:
    for i, line in enumerate(train):
        line_list = [int(k) for k in line[:-1].split("\t")]
        vertex_set.add(line_list[0]);
        for a in line_list[1:]:
            if a in sink_dict:
                sink_dict[a] += 1;
            else:
                sink_dict[a] = 1;
        if i % 2000 == 0:
            print(i);

print(len(sink_dict));
print(len(vertex_set));

n_sink_dict = {};
threshold = 10;

for i in sink_dict:
    if sink_dict[i] >= threshold:
        n_sink_dict[i] = sink_dict[i]
        
n_sink_set = set(n_sink_dict)
print(n_sink_set)
