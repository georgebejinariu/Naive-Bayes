# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import statistics
import math
import stat

N = 200
xgauss1 = [rd.gauss(0,1) for i in range(N)]
xgauss2 = [rd.gauss(0,1) for i in range(N)]


mu_1 = [2.5, 2.5]
sigma_1 = [[2, -0.8],
         [-0.8, 2]]


mu_2= [0,0]
sigma_2 = [[1,0],[0,1]]

gauss_2d_1 = np.random.multivariate_normal(mu_1, sigma_1, 100)


gauss_2d_2 = np.random.multivariate_normal(mu_2, sigma_2, 200)


x1Eval = gauss_2d_1

x2Eval = gauss_2d_2

figure = plt.figure(figsize=(10,3));ax = plt.subplot(1,3,1)

#ax.scatter(x1Eval, x2Eval,c='red',marker='x')





ax.set_title('New inputs')
ax.set_xlabel('x1')
ax.set_ylabel('x2')



def plot_r(x):
    vec_1 = []
    vec_2 = []
    for i in range(len(x)):
        vec_1.append(x[i][0])
        vec_2.append(x[i][1])
    return vec_1, vec_2

x_1,x_2 = plot_r(x1Eval)

y_1, y_2 = plot_r(x2Eval)



ax.scatter(x_1, x_2,c='red',marker='x')
ax.scatter(y_1, y_2,c='blue',marker='o')




print("The mean of the first gaussian is:", "[",np.mean(x_1), np.mean(x_2),"]")
print("The mean of second gaussian is:", '[',np.mean(y_1), np.mean(y_2),"]")
print("\n")
print("Covariance of the first gaussian \n",np.cov(x_1,x_2))
print("\n")
print("Covariance of the second gaussian \n",np.cov(y_1,y_2))




#Mean and standard deviasions

mean1_1 = np.mean(x_1)
mean1_2 = np.mean(x_2)

mean2_1 = np.mean(y_1)
mean2_2 = np.mean(y_2)


std1_1 = np.var(x_1)
std1_2 = np.var(x_2)

std2_1 = np.var(y_1)
std2_2 = np.var(y_2)


cov1 = np.cov(x_1,x_2)
cov2 = np.cov(y_1,y_2)

mean1 = [mean1_1, mean1_2]
mean2 = [mean2_1, mean2_2]






from math import sqrt
from math import pi
from math import exp
 
# Calculate the Gaussian probability distribution function for x

 
# # Test Gaussian PDF
# print(calculate_probability(1.0, 1.0, 1.0))
# print(calculate_probability(2.0, 1.0, 1.0))
# print(calculate_probability(0.0, 1.0, 1.0))



def compute_prob_1(x):
    prev =  (1/(sqrt(2*pi))*np.linalg.det(cov1))*exp((-1/2)*np.dot(np.dot(x-mean1,cov1),x-mean1))
    S = (1/3)*(1/(sqrt(2*pi))*np.linalg.det(cov1))*exp((-1/2)*np.dot(np.dot(x-mean1,cov1),x-mean1)) +(2/3)*(1/(sqrt(2*pi))*np.linalg.det(cov2))*exp((-1/2)*np.dot(np.dot(x-mean2,cov2),x-mean2))
    return (1/3)*prev/S


def compute_prob_2(x):
    prev =  (1/(sqrt(2*pi))*np.linalg.det(cov2))*exp((-1/2)*np.dot(np.dot(x-mean2,cov2),x-mean2))
    S = (1/3)*(1/(sqrt(2*pi))*np.linalg.det(cov1))*exp((-1/2)*np.dot(np.dot(x-mean1,cov1),x-mean1)) +(2/3)*(1/(sqrt(2*pi))*np.linalg.det(cov2))*exp((-1/2)*np.dot(np.dot(x-mean2,cov2),x-mean2))
    return (2/3)*prev/S



def N_Bayes(x):
    p1 = compute_prob_1(x)
    p2 = compute_prob_2(x)
    
    if p1>p2:
        return 0
    else:
        return 1
    
    



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


ax = plt.subplot(1,3,2)
ax.scatter(x1Eval_,x2Eval_,c=predxxclass,marker='x', cmap=mycolormap)
ax.set_title('MAP Bayes C.  on new inputs')
ax.set_xlabel('x1')
ax.set_ylabel('x2')


# ax = plt.subplot(1,3,3)
# Z=predxxclass
# ax.set_title('MAP Bayes C. Decision boudaries')

# Z = Z.reshape(x1Eval.shape)
# cm = plt.cm.RdBu
# ax.contourf(x1Eval_, x2Eval_, Z, cmap=cm, alpha=.8);
# ax.scatter(x1Eval_,x2Eval_,c='gray',marker='x')

