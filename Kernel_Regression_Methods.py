# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:22:16 2021
@author: brussell
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
# Create input data
N = 3; x = np.array([1, 2, 3]); y = np.array([1, 3, 2])
# Create polynomial kernel where p = polynomial order
p = 2; K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = (1 + x[i]*x[j])**p
# Invert kernel and create a
PW = 0.000001; I = np.identity(N)
Kinv = inv(K+PW*I); a = np.matmul(Kinv,y.T)
# Compute output
xstart = -1; xend = 4; ystart = -2; yend = 4; M = 50
x0 = np.linspace(xstart,xend,num = M)
ykp = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        ykp[i] += a[j]*(1 + x0[i]*x[j])**p
# Gaussian kernel regression
sigma = 10
K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = np.exp(-(x[i]-x[j])**2/(2*sigma**2))
# Create a
Kinv = inv(K+PW*I)
a = np.matmul(Kinv,y.T)
# Compute output
ykg = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        ykg[i] += a[j]*np.exp(-(x0[i]-x[j])**2/(2*sigma**2))
# tanh kernel regression
b = 0; c = 0.25;
K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = np.tanh(c*x[i]*x[j]+b)
# Create a
Kinv = inv(K+PW*I)
a = np.matmul(Kinv,y.T)
print(a)
# Compute output
ykt = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        ykt[i] += a[j]*np.tanh(c*x0[i]*x[j]+b)
# Logistic kernel regression
sigma = 0.25
K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = ((1+np.exp(-(x[i]*x[j]))/sigma))**-1
# Create a
Kinv = inv(K+PW*I)
a = np.matmul(Kinv,y.T)
# Compute output
yks = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        yks[i] += a[j]*((1+np.exp(-(x0[i]*x[j]))/sigma))**-1
# Generalized regression neural network
sigma = 2; one = np.ones(N)
ygrnn = np.zeros((M)); Kgrnn = np.identity(N);
for i in range(0,M):
    ygrnn[i] = 0.0
    for j in range(0,N):
        Kgrnn[j,j] = np.exp(-(x0[i]-x[j])**2/(2*sigma**2)) 
    Ksum = np.matmul(Kgrnn,one)
    num = np.matmul(y.T,Ksum)
    den = np.matmul(one.T,Ksum)
    ygrnn[i] = num/den
# Plot polynomial kernel regression
plt.figure(1)
plt.plot(x,y,'ko')
plt.plot(x0,ykp)
plt.title('Polynomial Kernel Regression : p = 3')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart,xend)
plt.ylim(ystart,yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot Gaussian kernel regression
plt.figure(2)
plt.plot(x,y,'ko')
plt.plot(x0,ykg)
plt.title('Gaussian Kernel Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot tanh kernel regression
plt.figure(3)
plt.plot(x,y,'ko')
plt.plot(x0,ykt)
plt.title('tanh Kernel Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot logistic kernel regression
plt.figure(4)
plt.plot(x,y,'ko')
plt.plot(x0,yks)
plt.title('Logistic Kernel Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot GRNN result
plt.figure(5)
plt.plot(x,y,'ko')
plt.plot(x0,ygrnn,'b')
plt.title('GRNN Result')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot all kernel regression methods
plt.figure(6)
plt.plot(x,y,'ko')
plt.plot(x0,ykp,'b')
plt.plot(x0,ykg,'r')
plt.plot(x0,ykt,'g')
plt.plot(x0,ygrnn,'k')
plt.title('Poly = blue, Gaussian = red, tanh = green, GRNN = black')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart, xend)
plt.ylim(ystart, yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
