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

train = np.matrix([feature_HAA, feature_HJC, feature_HRA]).T

train_labels = np.transpose([1 for x in range(len(HAA_train_positive))]
                        + [0 for x in range(len(HAA_train_negative))])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train, train_labels)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1)
clf.fit(X_train, Y_train)          

from sklearn.metrics import accuracy_score
Y_test_pred = clf.predict(X_test)
print(accuracy_score(Y_test, Y_test_pred))



# load the test data

#with np.load('test_features.npz') as tft:
#    HAA_test = tft['HAA']
#    HJC_test = tft['HJC']
#    HRA_test = tft['HRA']
    
#test_features = {'HAA': np.array(HAA_test),
#                 'HJC': np.array(HJC_test),
#                 'HRA': np.array(HRA_test)} 

#test = np.matrix([HAA_test, HJC_test, HRA_test]).T 

# print (test)
# print(test.shape)

# a = np.array(test)
# b = a.ravel()
# print(b)
# print(len(b))
# logsitic regression
