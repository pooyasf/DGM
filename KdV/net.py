#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:47:18 2020

@author: Pooya
"""

from libs import *


class Net(nn.Module):
    
    def __init__(self , NL , NN ):
        super(Net, self).__init__()
        
        self.NL = NL
        self.NN = NN
        
        self.fc_input = nn.Linear(2,self.NN)
        #torch.nn.init.xavier_uniform_(self.fc_input.weight)
        
        
        self.linears = nn.ModuleList([nn.Linear(self.NN, self.NN) for i in range(self.NL)])
        #for i, l in enumerate(self.linears):    
        #    torch.nn.init.xavier_uniform_(l.weight)
        
        
        self.fc_output = nn.Linear( self.NN , 1 )
        #torch.nn.init.xavier_uniform_(self.fc_output.weight)
 
        self.act = torch.exp #torch.tanh
        
    def forward(self, x):
        h = self.act( self.fc_input(x)  )
        
        
        for i, l in enumerate(self.linears):
            h = self.act( l(h) )
        
        out =            self.fc_output(h)
        
        return out 
    


class DGMNet(nn.Module):
    
    def __init__(self):
        super(DGMNet, self).__init__()
        self.S1_W = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.S1_W.weight)
        
        ### L=1
        self.Z1_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.Z1_U.weight)
        self.Z1_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.Z1_W.weight)
        self.G1_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.G1_U.weight)
        self.G1_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.G1_W.weight)
        self.R1_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.R1_U.weight)
        self.R1_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.R1_W.weight)
        self.H1_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.H1_U.weight)
        self.H1_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.H1_W.weight)
        
        ### L=2
        self.Z2_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.Z2_U.weight)
        self.Z2_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.Z2_W.weight)
        self.G2_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.G2_U.weight)
        self.G2_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.G2_W.weight)
        self.R2_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.R2_U.weight)
        self.R2_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.R2_W.weight)
        self.H2_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.H2_U.weight)
        self.H2_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.H2_W.weight)
        
        
        ### L=3
        self.Z3_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.Z3_U.weight)
        self.Z3_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.Z3_W.weight)
        self.G3_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.G3_U.weight)
        self.G3_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.G3_W.weight)
        self.R3_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.R3_U.weight)
        self.R3_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.R3_W.weight)
        self.H3_U = nn.Linear(2,50)
        torch.nn.init.xavier_uniform_(self.H3_U.weight)
        self.H3_W = nn.Linear(50,50)
        torch.nn.init.xavier_uniform_(self.H3_W.weight)
    
        
        self.f = nn.Linear( 50 , 1 )
        torch.nn.init.xavier_uniform_(self.f.weight)
        
        self.tanh = torch.tanh
        
    def forward(self, x):
        
        S1 = self.tanh( self.S1_W(x) )
        
        ### L=1
        Z1 = self.tanh( self.Z1_U( x ) + self.Z1_W( S1 ) )
        G1 = self.tanh( self.G1_U( x ) + self.G1_W( S1 ) )
        R1 = self.tanh( self.R1_U( x ) + self.R1_W( S1 ) )
        H1 = self.tanh( self.H1_U(x) + self.H1_W( torch.mul(S1, R1) )  )
        S2 = torch.mul( ( 1 - G1 ) , H1 ) + torch.mul( Z1 , S1 )
        
        ### L=2
        Z2 = self.tanh( self.Z2_U( x ) + self.Z2_W( S2 ) )
        G2 = self.tanh( self.G2_U( x ) + self.G2_W( S2 ) ) ## S1 or S2 ??
        R2 = self.tanh( self.R2_U( x ) + self.R2_W( S2 ) )
        H2 = self.tanh( self.H2_U(x) + self.H2_W( torch.mul(S2, R2) )  )
        S3 = torch.mul( ( 1 - G2 ) , H2 ) + torch.mul( Z2 , S2 )
        
        ### L=3
        Z3 = self.tanh( self.Z3_U( x ) + self.Z3_W( S3 ) )
        G3 = self.tanh( self.G3_U( x ) + self.G3_W( S3 ) ) ## S1 or S3 ??
        R3 = self.tanh( self.R3_U( x ) + self.R3_W( S3 ) )
        H3 = self.tanh( self.H3_U(x) + self.H3_W( torch.mul(S3, R3) )  )
        S4 = torch.mul( ( 1 - G3 ) , H3 ) + torch.mul( Z3 , S3 )
        
        ### Output layer
        f = self.f( S4 )
        
        return f