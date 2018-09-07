import numpy as np
import pandas as pd 

# load the training data
with np.load('new_positive_original.npz') as fp:
    HAA_train_positive = fp['HAA']
    HJC_train_positive = fp['HJC']
    HRA_train_positive = fp['HRA']
    SD_train_positive = fp['SD']

with np.load('new_negative_original.npz') as fn:
    HAA_train_negative = fn['HAA']
    HJC_train_negative = fn['HJC']
    HRA_train_negative = fn['HRA']
    SD_train_negative = fn['SD']

feature_HAA = list(HAA_train_positive) + list(HAA_train_negative)
feature_HJC = list(HJC_train_positive) + list(HJC_train_negative)
feature_HRA = list(HRA_train_positive) + list(HRA_train_negative)
feature_SD = list(SD_train_positive) + list(SD_train_negative)
train_features = {'HAA': np.array(feature_HAA),
                  'HJC': np.array(feature_HJC),
                  'HRA': np.array(feature_HRA),
                  'SD': np.array(feature_SD)}

X_train = np.matrix([feature_HJC]).T
# , feature_HJC, feature_HRA, feature_SD
Y_train = np.transpose([1 for x in range(len(HAA_train_positive))]
                        + [0 for x in range(len(HAA_train_negative))])

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(train, train_labels)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1)
clf.fit(X_train, Y_train)          

with np.load('new_test_original.npz') as tft:
    HAA_test = tft['HAA']
    HJC_test = tft['HJC']
    HRA_test = tft['HRA']
    SD_test = tft['SD']

test_features = {'HAA': np.array(HAA_test),
                 'HJC': np.array(HJC_test),
                 'HRA': np.array(HRA_test),
                 'SD': np.array(SD_test)}

test = np.matrix([HJC_test]).T                 
# , HJC_test, HRA_test, SD_test    
from sklearn.metrics import accuracy_score
test_pred = clf.predict_proba(test)

id_list = list(range(1, 2001))

result = []
for i in test_pred:
    result.append(i[1])
dataframe = pd.DataFrame({'Id':id_list, 'Prediction':result})
dataframe.to_csv("prediction.csv", index=False, sep=',')
