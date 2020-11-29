#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:55:10 2020

@author: Pooya
"""

from libs import *

## Sampling rectangular Domain

class Sampler():
    
    def __init__(self , net):
        
        ### Domain size

        self.MAX_T = np.pi
        self.MAX_X = np.pi
        ## for accept reject purpose!
        self.net = net

    def Sample(self , size = 2**8 ):
         
         te = self.MAX_T
         xe = self.MAX_X
         ts = 0
         xs = 0
         
         x = torch.cat(( torch.rand( [size,1] )*te  , torch.rand( [size,1] )*xe  ) , dim = 1 ).cuda()
         x_initial = torch.cat(( torch.zeros(size, 1)  , torch.rand( [size,1] )*xe  ) , dim = 1 ).cuda()
            
         x_boundry_start = torch.cat(( torch.rand( [size,1] )*te , torch.zeros(size, 1)  ) , dim = 1 ).cuda()
         x_boundry_end = torch.cat(( torch.rand( [size,1] )*te  , torch.zeros(size, 1)  ) , dim = 1 ).cuda()
         
         return x , x_initial , x_boundry_start , x_boundry_end