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
from sampler import *
from loss import *

#integration

net = Net( NL = 2 , NN = 30 )
net.to(torch.device("cuda:0"))  

heatloss = HeatLoss()
## providing sampler with net so it can accept/reject based on net and other criterions
heatsampler  = Sampler(net)    
#register_hook(net)
    
train = Train( net , heatloss , heatsampler , BATCH_SIZE = 2**8 , debug = True )
    
train.train( epoch = 2000 , lr = 0.001 )

train.plot_report()
train.plot_activation_mean()









