'''
Why option price with / without inequlity leads to same
solution?
why solution differs a little bit from
http://www.math.columbia.edu/~smirnov/options13.html


'''


import numpy as np
import matplotlib.pyplot as plt


################## Parameters:

    

d = 1
ep = 0.25
ru = 0.75
r = 0
c = 0.02
k = 1
T = 2
X0 = 1


## stable? How?
## dx should be a lot bigger than dt for stability!
dt = 0.001
dx = 0.01

(ep**2)*dt < dx**2

M = int(2*k/dx)
N = int(T/dt)

##################

ep_ = ( (d*ep*ep) + (d)*(d-1)*(ru*ep*ep) ) / (d*d)
mu_ = (r-c) - 0.5 * (ep_**2) + 0.5*(ep**2)

def g(x):
    return np.maximum(x,np.zeros( len(x) ).reshape(-1,1))


##################

alpha  = (1/4)*(ep_**2)*(dt/dx)
beta   = dt*r
theta  = (mu_/4) * dt


############ A

A = np.array( np.zeros((2*M+1)**2).reshape(2*M+1,2*M+1) )
A[0,0] = 1  
A[-1,-1] = 1

for i in range(1,2*M):
    j = i
    A[i,j-1]   = -theta*j + alpha*j
    A[i,j  ]   = -1 - 2 * alpha * j
    A[i,j+1]   = theta*j + alpha*j
    
Ai = np.linalg.inv(A)


############ Initial

x = np.array( np.zeros(2*M+1).reshape(-1,1) )
u = np.array( np.zeros((2*M+1)*N).reshape(N,2*M+1) )
u[0,:] = g( np.linspace(-k,k,2*M+1).reshape(-1,1) ).reshape(1,2*M+1)

########### b

b = np.array( np.zeros(2*M+1).reshape(-1,1) )
b[0]  = 0
b[-1] = k

for k in range(0,N-1):
    for j in range(1,2*M):
        b[j] = u[k,j-1]*(theta*j-alpha*j) + u[k,j]*(-1+2*alpha*j+beta) + u[k,j+1]*(-theta*j-alpha*j)
    x = Ai.dot(b)
    # check for arbitrage
    x = g(x)
    u[k+1,:] = x.reshape(1,2*M+1)


print( u[ u.shape[0]-1 , int(u.shape[1]/2) ])
#plt.plot(np.linspace(-k,k,2*M+1), b)
#plt.plot(np.linspace(-k,k,2*M+1), np.linspace(-k,k,2*M+1) )













