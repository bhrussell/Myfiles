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
mx = np.mean(x)
sx = np.std(x)
x = (x-mx)/sx
noise = 0.15*np.random.normal(0,1,10)
y = np.zeros((N))
for i in range(0, N):
    y[i] = np.sin(i*2*np.pi/N)+noise[i]+1;
# A fixed instance of the noisy sine wave, for comparison purposes
y = np.array([1.07,1.22,1.93,1.77,1.24,1.04,0.39,-0.10,0.41,0.29])
my = np.mean(y)
sy = np.std(y)
y = (y - my)/sy
p = 10
#print('Polynomial order:',p)
xstart = -2; xend = 2; ystart = -2; yend = 2; M = 50
#xstart = 0; xend = 11; ystart = -1; yend = 3; M = 50
x0 = np.linspace(xstart,xend,num = M)
# Compute regression coefficients
PW = 0.00001
I = np.identity(N)
# Polynomial kernel regression
K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = (1 + x[i]*x[j])**p
# Create a
Kinv = inv(K+PW*I)
a = np.matmul(Kinv,y.T)
# Compute output
ykp = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        ykp[i] += a[j]*(1 + x0[i]*x[j])**p
# Gaussian kernel regression
sigma = 0.5
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
b = 1; c = 1;
K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = np.tanh(c*x[i]*x[j]+b) 
# Create a
Kinv = inv(K+PW*I)
a = np.matmul(Kinv,y.T)
# Compute output
ykt = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        ykt[i] += a[j]*np.tanh(c*x0[i]*x[j]+b)
# Logistic kernel regression
b = 1; c = 1;
K = np.zeros((N,N)); 
for i in range(0,N):
    for j in range(0,N):
        K[i,j] = ((1+np.exp(x[i]*x[j])*c+b))**-1
# Create a
Kinv = inv(K+PW*I)
a = np.matmul(Kinv,y.T)
# Compute output
yks = np.zeros((M))
for i in range(0,M):
    for j in range(0,N):
        yks[i] += a[j]*((1+np.exp(x0[i]*x[j])*c+b))**-1
# Generalized regression neural network
sigma = 0.1; one = np.ones(N)
ygrnn = np.zeros((M))
for i in range(0,M):
    Kgrnn = np.identity(N); ygrnn[i] = 0.0
    for j in range(0,N):
        Kgrnn[j,j] = np.exp(-(x0[i]-x[j])**2/(2*sigma**2)) 
    Kone = np.matmul(Kgrnn,one)
    num = np.matmul(y.T,Kone)
    den = np.matmul(one.T,Kone)
    ygrnn[i] = num/den
# Plot polynomial kernel regression
plt.figure(1)
plt.plot(x,y,'ko')
plt.plot(x0,ykp)
plt.title('Polynomial Kernel Regression')
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
#plt.scatter(x,y,c ='black',s=100)
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
