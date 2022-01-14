# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 16:22:16 2021
@author: brussell
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
# Create input data and initialize arrays
N = 3; x = np.array([1, 2, 3]); y = np.array([1, 3, 2])
mu1 = 1; mu2 = 2; mu3 = 3; sigma =0.1;
one = np.ones(N); phi1 = np.zeros((N)); phi2 = np.zeros((N));
phi3 = np.zeros((N)); w = np.zeros((N))
xstart = 0; xend = 4; ystart = 0; yend = 4; M = 100
x0 = np.linspace(xstart,xend,num = M)
y0 = np.zeros((M))
y1 = np.zeros((M))
y2 = np.zeros((M))
# Single basis function
for i in range(0,N):
    phi1[i] = np.exp(-(x[i]-mu1)**2/(2*sigma**2))
phi = np.column_stack((one.T,phi1.T))
A = np.matmul(phi.T,phi); c = np.matmul(phi.T,y)
# Compute inverse and weights
PW = 0.000001; I = np.identity(2)
A_inv = inv(A+PW*I); w = np.matmul(A_inv,c) 
# Compute output
for i in range(0,M):
    phi01 = np.exp(-(x0[i]-mu1)**2/(2*sigma**2))
    y0[i] = w[0] + w[1]*phi01
# Two basis functions
for i in range(0,N):
    phi2[i] = np.exp(-(x[i]-mu2)**2/(2*sigma**2))
phi = np.column_stack((phi1.T,phi2.T))
A = np.matmul(phi.T,phi); c = np.matmul(phi.T,y)
# Compute inverse and weights
PW = 0.000001; I = np.identity(2)
A_inv = inv(A+PW*I); w = np.matmul(A_inv,c) 
# Compute output
for i in range(0,M):
    phi01 = np.exp(-(x0[i]-mu1)**2/(2*sigma**2))
    phi02 = np.exp(-(x0[i]-mu2)**2/(2*sigma**2))
    y1[i] = w[0]*phi01 + w[1]*phi02
# Three basis functions
for i in range(0,N):
    phi3[i] = np.exp(-(x[i]-mu3)**2/(2*sigma**2))
phi = np.column_stack((phi1.T,phi2.T,phi3.T))
A = np.matmul(phi.T,phi); A_inv = inv(A)
c = np.matmul(phi.T,y); w = np.matmul(A_inv,c) 
# Compute output
for i in range(0,M):
    phi01 = np.exp(-(x0[i]-mu1)**2/(2*sigma**2))
    phi02 = np.exp(-(x0[i]-mu2)**2/(2*sigma**2))
    phi03 = np.exp(-(x0[i]-mu3)**2/(2*sigma**2))
    y2[i] = w[0]*phi01 + w[1]*phi02+w[2]*phi03
# Plot one basis function
plt.figure(1)
plt.plot(x,y,'ko')
plt.plot(x0,y0)
plt.title('Single Gaussian Basis Function')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart,xend)
plt.ylim(ystart,yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot two basis functions
plt.figure(2)
plt.plot(x,y,'ko')
plt.plot(x0,y1)
plt.title('Two Gaussian Basis Functions')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart,xend)
plt.ylim(ystart,yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()
# Plot three basis functions
plt.figure(2)
plt.plot(x,y,'ko')
plt.plot(x0,y2)
plt.title('Three Gaussian Basis Functions, sigma = 1')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(xstart,xend)
plt.ylim(ystart,yend)
plt.grid(color='k', linestyle='--', linewidth=1)
plt.show()

