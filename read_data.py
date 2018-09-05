import numpy as np
pairs = []
with open("../data/train.txt") as file:
    count = 0
    for i, line in enumerate(file):
        line_list = [int(k) for k in line[:-1].split("\t")]
        if len(line_list) > 1:
            source = line_list[0]
            for l in line_list[1:]:
                pairs.append((source, l))
                count += 1
                if i % 1000 == 0:
                    print("line of file: ", i, "num of pairs: ", count)

np.savez_compressed("original_pairs", pairs=pairs)
