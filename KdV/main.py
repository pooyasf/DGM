#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:49:51 2020

@author: Pooya    

"""

from libs import *
from train import *
from net import *
#import hook
from kdv import *


#net = Net( NL = 10 , NN = 50 )
net = DGMNet()
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
kdvequation = KDV(net)
#register_hook(net)
    
train = Train( net , kdvequation , BATCH_SIZE = 2**8 , debug = True )
    
train.train( epoch = 10000 , lr = 0.00005 )

train.plot_report()
train.plot_activation_mean()



#%%% SURFACE PLOT

MAX_X = 2
MAX_T = 1.145

y_range = np.linspace(-MAX_X/2, MAX_X/2, 40, dtype=np.float)
x_range = np.linspace( -MAX_T/2 , MAX_T/2 , 40, dtype=np.float)

data = np.empty((2,1))

Z = []
for _x in x_range:
    data[0] = _x
    for _y in y_range:
        data[1] = _y
        indata = torch.Tensor(data.reshape(1,-1)).cuda()
        Zdata = net(indata).detach().cpu()
        Z.append(Zdata)


_X, _Y = np.meshgrid(x_range, y_range, indexing='ij')

Z_surface = np.reshape(Z, (x_range.shape[0], y_range.shape[0]))

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface( _X, _Y, Z_surface,  cmap=cm.YlOrBr_r, edgecolor='gray', linewidth=0.004, antialiased=False)
plt.show()


#%%

x_terminal = torch.cat(( torch.zeros(len(y_range), 1) + MAX_T/2 , torch.tensor(y_range).float().reshape(-1,1) ) , dim = 1 ).cuda()
x_initial  = torch.cat(( torch.zeros(len(y_range), 1) - MAX_T/2 , torch.tensor(y_range).float().reshape(-1,1) ) , dim = 1 ).cuda()
fig = plt.figure()
plt.plot( y_range , net( x_initial  ).cpu().detach() )
plt.plot( y_range , net( x_terminal  ).cpu().detach() )





