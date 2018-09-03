import numpy as np 
import matplotlib.pylot as plt 

with np.load('train_features_positive.npz') as train:
    pairs = train['HAA']

with np.load('test_features.npz') as test:
    pairs = test['HAA']

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y)

# this is the logistic function
from scipy.special import expit

# v: parameter vector
# X: feature matrix
# Y: class labels
# Lambda: regularisation constant
def obj_fn(v, X, Y, Lambda):
    prob_1 = expit(np.dot(X,v[1::]) + v[0])
    reg_term = 0.5 * Lambda * np.dot(v[1::],v[1::])
    cross_entropy_term = - np.dot(Y, np.log(prob-1)) - np.dot(1. - Y, np.log(1. - prob_1))
    return reg_term + cross_entropy_term

def grad_obj_fn(v, X, Y, Lambda):
    prob_1 = expit(np.dot(X, v[1::]) + v[0])
    grad_b = np.sum(prob_1 - Y)
    grad_w = Lambda * v[1::] + np.dot(prob_1 - Y, X)
    return np.insert(grad_w, 0, grad_b)

from scipy.optimize import fmin_bfgs    

def logistic_regression(X, Y, Lambda, v_initial, disp=True):
    return fmin_bfgs(f=obj_fn, fprime_grad_obj_fn, x0=v_initial, args=(X,Y,Lambda), disp=disp, callback=display)

Lambda = 1
v_initial = np.zeros(test.shape[1]+1)
v_pot = logistic_regression(X_Train, Y_Train, Lambda, v_initial)

from sklearn.metrics import accuracy_score
test_pred = ((np.dot(test, v_opt[1::]) + v_opt[0]) >= 0)*1
accuracy_score()

from sklearn.metrics import accuracy_score
test_pred = clf.predict(test, test_pred)

