import numpy as np
import matplotlib.pyplot as plt

'''
for stable answer:
\Delta t \leq \frac{2\Delta x^3}{3\sqrt{3}}
'''

delta = 0.022
X = 2
TB = 1/np.pi
TR = 30.4*TB

N = 64
M = 778911

h = 1/N        ## x axis
k = 1.47e-06   ## time

def uc(j,i):
    return u[j,i%(2*N)]

u = np.array( np.zeros( (2*N)*M ).reshape(M,2*N) )



for i in range(0,2*N):
    u[0,i] = np.cos(np.pi*i*h)

j = 0

## forward euler for first step
for i in range(0,2*N):
    u[j+1,i] = uc(j,i) - (1/6)*(k/h)*( uc(j,i+1) + uc(j,i) + uc(j,i-1) )*( uc(j,i+1) - uc(j,i-1) ) \
               - (1/2)*(delta**2)*(k/(h**3))*( uc(j,i+2) - 2*uc(j,i+1) + 2*uc(j,i-1) - uc(j,i-2) )

## same scheme but just approximate uc(j-1,i) = uc(j,i)
#for i in range(0,2*N):
#    u[j+1,i] = uc(j,i) - (1/3)*(k/h)*( uc(j,i+1) + uc(j,i) + uc(j,i-1) )*( uc(j,i+1) - uc(j,i-1) ) \
#                   - (delta**2)*(k/(h**3))*( uc(j,i+2) - 2*uc(j,i+1) + 2*uc(j,i-1) - uc(j,i-2) )
                                                 

for j in range(1,M-1):
    
    for i in range(0,2*N):
        u[j+1,i] = uc(j-1,i) - (1/3)*(k/h)*( uc(j,i+1) + uc(j,i) + uc(j,i-1) )*( uc(j,i+1) - uc(j,i-1) ) \
                   - (delta**2)*(k/(h**3))*( uc(j,i+2) - 2*uc(j,i+1) + 2*uc(j,i-1) - uc(j,i-2) )
                                                 

plt.plot(u[M-1,])


np.savetxt("kdv.csv", u, delimiter=",")
