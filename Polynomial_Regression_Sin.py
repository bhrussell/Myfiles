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
N = 10
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Compute one period of a sine wave corrupted with random noise
# Note that the result will be different on each pass
noise = 0.15*np.random.normal(0,1,10)
y = np.zeros((N))
for i in range(0, N):
    y[i] = np.sin(i*2*np.pi/N)+noise[i]+1;
# A fixed instance of the noisy sine wave, for comparison purposes
y = np.array([1.07,1.22,1.93,1.77,1.24,1.04,0.39,-0.10,0.41,0.29])
# If primal = 1 perform primal regression
# If primal = 0 perform dual regression
primal = 0
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
    Cinv = inv(C+PW*I)
    Xgi = np.matmul(Cinv,X.T)
    print('Primal Inverse:')
    print(Xgi)
    w = np.matmul(Xgi,y.T)
if primal == 0:
    C = np.matmul(X,X.T)
    I = np.identity(N)
    Cinv = inv(C+PW*I)
    Xgi = np.matmul(X.T,Cinv)
    w = np.matmul(Xgi,y.T)
    print('Dual Inverse:')
    print(Xgi)
print('Regression coefficients:')
print(w)
# Create regression fit
# Define start and end of x and y axes
xstart = 0; xend = 11; ystart = -0.5; yend = 2.5; M = 100
x0 = np.linspace(xstart,xend,M)
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
plt.plot(x,y,'ko')
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
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()