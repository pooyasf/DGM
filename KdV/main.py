#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:49:51 2020

@author: Pooya    
/home/berdpen/Documents/University/ThesisRazvaan/GITHUB/DGM/KdV
"""

from libs import *
from train import *
from net import *
#import hook
from kdv import *


net = Net( NL = 6 , NN = 16 )
#net = DividedNet()
#net = DynaNet()
#net = DGMNet()
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
kdvequation = KDV(net)
#register_hook(net)

train = Train( net , kdvequation , BATCH_SIZE = 2**7 , debug = True )

#%%%

train.train( epoch = 1000 , lr = 0.01 )
train.train( epoch = 5000 , lr = 0.001 )
train.train( epoch = 1000 , lr = 0.0005 )
train.train( epoch = 1000 , lr = 0.001 )
train.train( epoch = 5000 , lr = 0.0001 )
train.train( epoch = 2000 , lr = 0.0005 )
train.train( epoch = 5000 , lr = 0.0001 )

#%%

net.SWITCH = 0

#%%

net.fc_input.weight.requires_grad = True
net.fc_input.bias.requires_grad = True

net.SWITCH = 1

#%%

print(net.SWITCH)

net.l1.weight.requires_grad = True
net.l1.bias.requires_grad = True


net.SWITCH = 2

print(net.SWITCH)

#%%


train.plot_report()
train.plot_activation_mean()



#%%% SURFACE PLOT

MAX_X = 2
MAX_T = 1.145

y_range = np.linspace(-MAX_X/2, MAX_X/2, 100, dtype=np.float)
x_range = np.linspace( -MAX_T/2 , MAX_T/2 , 100, dtype=np.float)



x_terminal = torch.cat(( torch.zeros(len(y_range), 1) + MAX_T/2 , torch.tensor(y_range).float().reshape(-1,1) ) , dim = 1 ).cuda()
x_initial  = torch.cat(( torch.zeros(len(y_range), 1) - MAX_T/2 , torch.tensor(y_range).float().reshape(-1,1) ) , dim = 1 ).cuda()
fig = plt.figure()
plt.plot( y_range , net( x_initial  ).cpu().detach() )
plt.plot( y_range , net( x_terminal  ).cpu().detach() )

#%%

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
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('Wave')
plt.show()



#%% save


torch.save(net.state_dict(), './modelMLP_3_128')


#%%

net.input_t.weight.requires_grad = False
net.input_t.bias.requires_grad = False

net.t_1.weight.requires_grad = False
net.t_1.bias.requires_grad = False

net.t_out.weight.requires_grad = False
net.t_out.bias.requires_grad = False


net.input_x.weight.requires_grad = False
net.input_x.bias.requires_grad = False

net.x_1.weight.requires_grad = False
net.x_1.bias.requires_grad = False

net.x_out.weight.requires_grad = False
net.x_out.bias.requires_grad = False




#%%


for param in net.parameters():
    print(param)



