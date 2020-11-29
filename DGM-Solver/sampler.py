#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:55:10 2020

@author: Pooya
"""

from libs import *

### Domain size

MAX_T = np.pi
MAX_X = np.pi


## Sampling Domain rectangular

def Sampler( ts = 0 , te = MAX_T , xs = 0 , xe = MAX_X , size = 2**8 ):
     
     x = torch.cat(( torch.rand( [size,1] )*te  , torch.rand( [size,1] )*xe  ) , dim = 1 ).cuda()
     x_initial = torch.cat(( torch.zeros(size, 1)  , torch.rand( [size,1] )*xe  ) , dim = 1 ).cuda()
        
     x_boundry_start = torch.cat(( torch.rand( [size,1] )*te , torch.zeros(size, 1)  ) , dim = 1 ).cuda()
     x_boundry_end = torch.cat(( torch.rand( [size,1] )*te  , torch.zeros(size, 1)  ) , dim = 1 ).cuda()
     
     return x , x_initial , x_boundry_start , x_boundry_end