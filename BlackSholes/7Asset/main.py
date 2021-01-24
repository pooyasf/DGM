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
from blacksholes import *

#integration

net = Net( NL = 3 , NN = 30 )
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
bsequation = BlackSholes(net)
#register_hook(net)
    
train = Train( net , bsequation , BATCH_SIZE = 2**9 , debug = True )

#%%   
 
train.train( epoch = 3000 , lr = 0.001 )


#%%

train.plot_report()
train.plot_activation_mean()



#%%%

print( 'Value at 0' , net( torch.tensor( [ 0. , 1. , 1. , 1. ] ).cuda() ) )



#%% save


torch.save(net.state_dict(), './model3Assets')

#%%

net = TheModelClass(*args, **kwargs)
net.load_state_dict(torch.load('./modelmodel3Assets'))
net.eval()




