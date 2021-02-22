#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:49:51 2020

@author: Pooya


"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from libs import *
from train import *
from net import *
#import hook
from heat import *

#integration

net = Net( NL = 2 , NN = 20 )
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
heatequation = Heat(net)
#register_hook(net)
    
train = Train( net , heatequation , BATCH_SIZE = 2**8 , debug = True )

#%%
  
train.train( epoch = 3000 , lr = 0.0001 )


#%%
train.plot_report()
train.plot_activation_mean()



#%% PLOTS

MAX_T = np.pi
MAX_X = np.pi

t_range = np.linspace(0, MAX_T , 100, dtype=np.float)
x_range = np.linspace(0, MAX_X , 100, dtype=np.float)

_T, _X = np.meshgrid(t_range, x_range, indexing='ij')


x = torch.tensor( np.concatenate( (_T.reshape(-1,1) , _X.reshape(-1,1)) , axis = 1 ) )
x = Variable(x , requires_grad = True).cuda().float()

tl , dl , il , bl = heatequation.criterion( x , x , x , x )
Z_surface = torch.reshape(tl, (t_range.shape[0], x_range.shape[0]))

fig, ax = plt.subplots()
CS = ax.contour(_T , _X , Z_surface.cpu().detach() , levels = 10)
ax.clabel(CS, inline=1, fontsize=8)
CB = fig.colorbar(CS, shrink=0.8, extend='both')
ax.set_title(' differential operators error ')
ax.set_xlabel(' t ', fontsize=10)
ax.set_ylabel(' x ', fontsize=10)

z = net(x)
Z_surface = torch.reshape(z, (t_range.shape[0], x_range.shape[0]))

fig, ax = plt.subplots()
CS = ax.contour(_T , _X , Z_surface.cpu().detach() , levels = 10)
ax.clabel(CS, inline=1, fontsize=8)
CB = fig.colorbar(CS, shrink=0.8, extend='both')
ax.set_title(' solution ')
ax.set_xlabel(' t ', fontsize=10)
ax.set_ylabel(' x ', fontsize=10)

z = net(x).cpu().detach() - heatequation.exact_solution( _T.reshape(-1,1) , _X.reshape(-1,1) )
mse_error = torch.mean( z**2 )
Z_surface = torch.reshape(z, (t_range.shape[0], x_range.shape[0]))

fig, ax = plt.subplots()
CS = ax.contour(_T , _X , Z_surface.cpu().detach() , levels = 10)
ax.clabel(CS, inline=1, fontsize=8)
CB = fig.colorbar(CS, shrink=0.8, extend='both')
ax.set_title(' net(x) - exact solution \n MSE: %f ' %mse_error.item())
ax.set_xlabel(' t ', fontsize=10)
ax.set_ylabel(' x ', fontsize=10)

#%%

x , x_initial , x_boundry_0 , x_boundry_pi = heatequation.sample( ts = 2 , te = 3 , xs = 2 , xe = 3 , size = 2**12 )

out = net(x)

train.hooks

#%%

jet= plt.get_cmap('jet')
colors = iter(jet(np.linspace(0,1,10)))
fig, ax = plt.subplots()
            
for i in train.hooks:
    #print(train.hooks[i].reshape(1,-1))
    ax.hist( train.hooks[i].reshape(1,-1).cpu().detach() , label= i , color=next(colors) ,\
            bins = np.linspace( -4 , 4 , 20) , histtype = 'step' , log = True , density = True )
fig.suptitle('Layers activation hist at end of training', fontsize=10)
leg = ax.legend();



#%% Running for different archs



# L_2 = np.zeros((30,1) , dtype = np.float)
# L_inf = np.zeros((30,1) , dtype = np.float)

# for nn in range(5,30) :

#     net = Net( NL = 1 , NN = nn )
#     net.to(torch.device("cuda:0"))  
    
#     #register_hook(net)
    
#     train = Train( net , BATCH_SIZE = 2**8 , debug = True )
    
#     train.train( epoch = 1000 , lr = 0.001 )
    
    
#     t_range = np.linspace(0, MAX_T , 100, dtype=np.float)
#     x_range = np.linspace(0, MAX_X , 100, dtype=np.float)

#     _T, _X = np.meshgrid(t_range, x_range, indexing='ij')
#     x = torch.tensor( np.concatenate( (_T.reshape(-1,1) , _X.reshape(-1,1)) , axis = 1 ) )
#     x = Variable(x , requires_grad = True).cuda().float()

#     z = net(x).cpu().detach() - exact_solution( _T.reshape(-1,1) , _X.reshape(-1,1) )
    
    
    
#     L_2[nn]    = torch.mean( z**2 )
#     L_inf[nn]  = torch.max( z**2 )
    
#     print(nn)



# fig, ax = plt.subplots()
# ax.plot(range(5,30),L_2[5:30] , '--r' ,  label=' L2 ')
# ax.plot(range(5,30),L_inf[5:30] ,  label=' Sup Norm ')
# ax.legend()
# ax.set_title(' One layer Neural Net ')
# ax.set_xlabel(' number of neurons ', fontsize=10)
# ax.set_ylabel(' error ', fontsize=10)





