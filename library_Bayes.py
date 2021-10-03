# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 11:30:41 2021

@author: georg
"""
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics





mu_1 = [2.5, 2.5]
sigma_1 = [[2, -0.8],
         [-0.8, 2]]


mu_2= [0,0]
sigma_2 = [[1,0],[0,1]]

X1= np.random.multivariate_normal(mu_1, sigma_1, 100)


X2= np.random.multivariate_normal(mu_2, sigma_2, 200)


def plot_r(x):
    vec_1 = []
    vec_2 = []
    for i in range(len(x)):
        vec_1.append(x[i][0])
        vec_2.append(x[i][1])
    return vec_1, vec_2

x_1,x_2 = plot_r(X1)

y_1, y_2 = plot_r(X2)


figure = plt.figure(figsize=(10,3));ax = plt.subplot(1,3,1)

#ax.scatter(x1Eval, x2Eval,c='red',marker='x')

ax.set_title('New inputs')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

ax.scatter(x_1, x_2,c='blue',marker='x')
ax.scatter(y_1, y_2,c='red',marker='o')


x1_min, x1_max = min(min(x_1),min(y_1)), max(max(x_1),max(y_1))
x2_min, x2_max = min(min(x_2),min(y_2)), max(max(x_2),max(y_2))
Neval = 15
h1 = (x1_max-x1_min)/Neval; h2 = (x2_max-x2_min)/Neval
x1Eval_,x2Eval_ = np.meshgrid(np.arange(x1_min, x1_max, h1),np.arange(x2_min, x2_max,h2)); 

figure = plt.figure(figsize=(10,3));
ax = plt.subplot(1,3,1)
ax.scatter(x1Eval_,x2Eval_,c='gray',marker='x')
ax.set_title('New inputs')
ax.set_xlabel('x1')
ax.set_ylabel('x2')

Mat = np.column_stack((x1Eval_,x2Eval_))




X = np.concatenate((X1,X2))
y = np.concatenate(([0 for i in range(100)],[1 for i in range(200)]))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# print("Number of mislabeled points out of a total %d points : %d"
#           % (X_test.shape[0], (y_test != y_pred).sum()))

gnb = GaussianNB();
gnbfit = gnb.fit(X,y)
y_pred = gnbfit.predict(X)





