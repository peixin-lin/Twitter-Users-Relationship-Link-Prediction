import numpy as np 
import matplotlib.pylot as plt 

# load the training data
with np.load('get_train_set_positive.npz') as fp:
    HAA_train_positive = fp['HAA']
    HJC_train_positive = fp['HJC']
    HRA_train_positive = fp['HRA']

with np.load('get_train_set_negative.npz') as fn:
    HAA_train_negative = fn['HAA']
    HJC_train_negative = fn['HJC']
    HRA_train_negative = fn['HRA']

feature_HAA = list(HAA_train_positive) + list(HAA_train_negative)
feature_HJC = list(HJC_train_positive) + list(HJC_train_negative)
feature_HRA = list(HRA_train_positive) + list(HRA_train_negative)
train_features = {'HAA': np.array(feature_HAA),
                  'HJC': np.array(feature_HJC),
                  'HRA': np.array(feature_HRA)}

train_labels = np.array([1 for x in range(len(HAA_train_positive))]
                        + [0 for x in range(len(HAA_train_negative))])

# load the test data
with np.load('test_features.npz') as tft:
    HAA_test = tft['HAA']
    HJC_test = tft['HJC']
    HRA_test = tft['HRA']

test_features = {'HAA': np.array(HAA_test),
                 'HJC': np.array(HJC_test),
                 'HRA': np.array(HRA_test)}

# logsitic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1)
clf.fit(train_features, train_labels)          

from sklearn.metrics import accuracy_score
test_pred = clf.predict(test_features)
accuracy_score(test_features, test_pred)