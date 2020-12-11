#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:49:51 2020

@author: Pooya

TO DO:
    
    1- Histogram of weigths gradients
    2- Std of activation functions
    3- Compare result with exact solution (Done)
    4- save result to folder
    5- compare result with different net archs ( Done , but not diff. depth! )
    6 - Save weights to file for furthure analysis
    

"""

from libs import *
from train import *
from net import *
#import hook
from heat import *

#integration

net = Net( NL = 2 , NN = 30 )
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
heatequation = Heat(net)
#register_hook(net)
    
train = Train( net , heatequation , BATCH_SIZE = 2**8 , debug = True )
    
train.train( epoch = 200 , lr = 0.001 )

train.plot_report()
train.plot_activation_mean()



#%%% SURFACE PLOT


y_range = np.linspace(0, MAX_X, 40, dtype=np.float)
x_range = np.linspace(0, MAX_T, 40, dtype=np.float)

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

x_terminal = torch.cat(( torch.zeros(len(y_range), 1) + MAX_T , torch.tensor(y_range).float().reshape(-1,1) ) , dim = 1 ).cuda()

plt.plot( net( x_terminal  ).cpu().detach() )





