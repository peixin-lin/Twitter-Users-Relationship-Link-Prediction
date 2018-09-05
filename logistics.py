import numpy as np 

# load the training data
with np.load('features_positive.npz') as fp:
    HAA_train_positive = fp['HAA']
    HJC_train_positive = fp['HJC']
    HRA_train_positive = fp['HRA']

with np.load('train_features_negative.npz') as fn:
    HAA_train_negative = fn['HAA']
    HJC_train_negative = fn['HJC']
    HRA_train_negative = fn['HRA']

feature_HAA = list(HAA_train_positive) + list(HAA_train_negative)
feature_HJC = list(HJC_train_positive) + list(HJC_train_negative)
feature_HRA = list(HRA_train_positive) + list(HRA_train_negative)
train_features = {'HAA': np.array(feature_HAA),
                  'HJC': np.array(feature_HJC),
                  'HRA': np.array(feature_HRA)}

# np.column_stack(feature_HAA, feature_HJC, feature_HRA)
X = np.matrix([feature_HAA, feature_HJC, feature_HRA]).T
print(X.shape)

# X.T = np.transpose(feature_HAA, feature_HJC, feature_HRA)

train_labels = np.transpose([1 for x in range(len(HAA_train_positive))]
                        + [0 for x in range(len(HAA_train_negative))])

# print(train_labels.shape)
# load the test data
with np.load('test_features.npz') as tft:
    HAA_test = tft['HAA']
    HJC_test = tft['HJC']
    HRA_test = tft['HRA']
    
test_features = {'HAA': np.array(HAA_test),
                 'HJC': np.array(HJC_test),
                 'HRA': np.array(HRA_test)} 

Y = np.matrix([HAA_test, HJC_test, HRA_test]).T 
print(Y.shape)

# logsitic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1)
clf.fit(X, train_labels)          

from sklearn.metrics import accuracy_score
test_pred = clf.predict(Y)
accuracy_score(Y, test_pred.round(), normalize=False)

# print(test_pred)
# for x in test_pred:
#     print(test_pred[])

