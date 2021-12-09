# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:22:16 2021
@author: brussell
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
#from numpy.linalg import det
# Create input data
N = 3
x = np.array([1, 2, 3])
y = np.array([1, 3, 2])
# If primal = 1 perform primal regression
# If primal = 0 perform dual regression
primal = 1
print('Solution (primal = 1, dual = 0):',primal)
p = 3
print('Polynomial order:',p)
# Compute regression coefficients
PW = 0.000001
one = np.ones(N)
x_square = np.power(x,2)
x_cube = np.power(x,3)
if p == 1:
    X = np.column_stack((one.T,x.T))
if p == 2:
    X = np.column_stack((one.T,x.T,x_square.T))
if p == 3:
    X = np.column_stack((one.T,x.T,x_square.T,x_cube.T))
if primal == 1:
    C = np.matmul(X.T,X)
    I = np.identity(p+1)
    Cinv = np.matmul(inv(C+PW*I),X.T)
    print('Primal Inverse:')
    print(Cinv)
    w = np.matmul(Cinv,y.T)
if primal == 0:
    C = np.matmul(X,X.T)
    I = np.identity(N)
    C1 = inv(C+PW*I)
    Cinv = np.matmul(X.T,C1)
    print('Dual Inverse:')
    print(Cinv)
    w = np.matmul(Cinv,y.T)
print('Regression coefficients:')
print(w)
# Create regression fit
# Define start and end of x and y axes
start = 0; end = 4
x0 = np.linspace(start,end,num = 100)
x0_square = np.power(x0,2)
x0_cube = np.power(x0,3)
if p == 1:
    y0 = w[0] + w[1]*x0
if p == 2:
   y0 = w[0] + w[1]*x0 + w[2]*x0_square
if p == 3:
    y0 = w[0] + w[1]*x0 + w[2]*x0_square + w[3]*x0_cube    
# Plot regression
plt.figure(1)
plt.scatter(x,y,c ='black',s=100)
plt.plot(x0,y0)
if p == 1 and primal == 1:
    plt.title('Linear Primal Regression')
if p == 1 and primal == 0:
    plt.title('Linear Dual Regression')
if p == 2 and primal == 1:
    plt.title('Quadratic Primal Regression')
if p == 2 and primal == 0:
    plt.title('Quadratic Dual Regression')
if p == 3 and primal == 1:
    plt.title('Cubic Primal Regression')
if p == 3 and primal == 0:
    plt.title('Cubic Dual Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(start, end)
plt.ylim(start, end)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()