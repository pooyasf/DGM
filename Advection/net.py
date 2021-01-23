#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:47:18 2020

@author: Pooya
"""

from libs import *


class Net(nn.Module):
    
    def __init__(self , NL = 2 , NN = 10 ):
        super(Net, self).__init__()
        
        self.NL = NL
        self.NN = NN
        
        self.fc_input = nn.Linear( 1 , self.NN )
        #torch.nn.init.xavier_uniform_(self.fc_input.weight)
        
        
        
        self.linears = nn.ModuleList([nn.Linear(self.NN, self.NN) for i in range(self.NL)])
        #for i, l in enumerate(self.linears):    
        #    torch.nn.init.xavier_uniform_(l.weight)
        
        
        self.fc_output = nn.Linear(self.NN,1)
        #torch.nn.init.xavier_uniform_(self.fc_output.weight)
 
        
        self.act = torch.tanh
        
    def forward(self, x):
        h = self.act( self.fc_input(x)  )
        
        
        for i, l in enumerate(self.linears):
            h = self.act( l(h) )
        
        out =            self.fc_output(h)
        
        return out 
    
