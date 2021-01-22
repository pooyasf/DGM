#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:06:34 2020

@author: berdpen

It doesn`t work!
Maybe because of stability issue
or Scheme is incorrect from pdf file


"""

import numpy as np
import matplotlib.pyplot as plt


sigma = 0.25
r = 0.
c = 0.02
ru = 0.75
d = 1 
k = 1

sigma_ = np.sqrt( (d*(sigma**2) + d*(d-1)*ru*(sigma**2))/(d**2) )
mu_    = (r-c) - 0.5 * (sigma_**2) + 0.5 * (sigma**2)

## Just even numbers
N = 16 ## price
M = 1600 ## time

T = 2
S_MAX = 2
S_MIN = 0

dt = T/M
dx = (S_MAX-S_MIN)/N

print(dt/(dx**2))

v1 = dt/(dx**2)
v2 = dt/dx


# 1 =< n =< N-1
A = 0.5*v1* (np.arange(0,N+1,1).reshape(-1,1) - N/2 ) + 0.25*v2*0.5*(sigma_*sigma_)* (np.arange(0,N+1,1).reshape(-1,1) - N/2 )
B = -v1*mu_* (np.arange(0,N+1,1).reshape(-1,1) - N/2 )
C = 0.5*v1*mu_* (np.arange(0,N+1,1).reshape(-1,1) - N/2 ) - 0.25*v2*0.5*(sigma_*sigma_)* (np.arange(0,N+1,1).reshape(-1,1) - N/2 )
##


M_L = np.array( np.zeros( (N-1)**2 ).reshape(N-1,N-1) )

M_L[0,0] = 1 + B[1]
M_L[0,1] = C[1]

for n in range(2,N-1):
    M_L[n-1,0+n-2] = A[n]
    M_L[n-1,1+n-2] = 1 + B[n]
    M_L[n-1,2+n-2] = C[n]

M_L[N-2,N-3] = A[N-1]
M_L[N-2,N-2]   = 1 + B[N-1]



M_R = np.array( np.zeros( (N-1)**2 ).reshape(N-1,N-1) )

M_R[0,0] = 1 - B[1]
M_R[0,1] = -C[1]

for n in range(2,N-1):
    M_R[n-1,0+n-2] = -A[n]
    M_R[n-1,1+n-2] = 1 - B[n]
    M_R[n-1,2+n-2] = -C[n]

M_R[N-2,N-3] = -A[N-1]
M_R[N-2,N-2]   = 1 - B[N-1]


R_L = np.array( np.zeros( (N-1) ).reshape(-1,1) )
R_R = np.array( np.zeros( (N-1) ).reshape(-1,1) )

R_L[0] = 0*A[1]
R_L[0] = S_MAX*C[N-1]


R_R[0] = -0*A[1]
R_R[0] = -1*C[N-1]


V_NOW  = np.array( np.zeros( (N-1) ).reshape(-1,1) )
V_NEXT = np.array( np.zeros( (N-1) ).reshape(-1,1) )

def g(x):
    return np.maximum( x , np.zeros( len(x) ).reshape(-1,1) )

V_NOW = g(np.arange(S_MIN,S_MAX,dx).reshape(-1,1))[1:]


for m in range(M):
    V_NEXT = np.linalg.inv(M_L).dot(  M_R.dot(V_NOW) + R_R - R_L  )
    V_NOW  = np.maximum( g(np.arange(S_MIN,S_MAX,dx).reshape(-1,1))[1:] , V_NEXT )





