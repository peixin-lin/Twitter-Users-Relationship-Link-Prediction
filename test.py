import numpy as np

with np.load('new_test.npz') as tft:
    HAA_test = tft['HAA']
    HJC_test = tft['HJC']
    HRA_test = tft['HRA']
    SD_test = tft['SD']

print(np.var(HAA_test))
print(np.var(HJC_test))
print(np.var(HRA_test))
print(np.var(SD_test))