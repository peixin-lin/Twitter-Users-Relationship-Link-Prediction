import numpy as np
# import tensorflow as tf
#
# a = tf.Variable([0.1, 0.5])
# b = tf.Variable([0.1, 0.5])
#
# auc = tf.contrib.metrics.streaming_auc(a, b)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# sess.run(tf.initialize_local_variables()) # try commenting this line and you'll get the error
# train_auc = sess.run(auc)
#
# print(train_auc)

with np.load('new_negative_original.npz') as tft:
    HAA_test = tft['HAA']
    HJC_test = tft['HJC']
    HRA_test = tft['HRA']
    SD_test = tft['SD']

print(len(HAA_test))
print(np.var(HAA_test))
print(np.var(HJC_test))
print(np.var(HRA_test))
print(np.var(SD_test))
