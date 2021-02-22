#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:48:58 2020

@author: Pooya
"""

from libs import *
from hook import *


class Train():

    
    def __init__(self , net , heatequation , BATCH_SIZE , debug = False):
        
        
        self.history_mean_hooks = []
        
        self.history_tl = []
        self.history_dl = []
        self.history_il = []
        self.history_bl = []
              
        self.BATCH_SIZE = BATCH_SIZE
        self.net = net
        self.model = heatequation
        
        self.debug = debug
        
        if self.debug == True:
            self.hooks = {}
            
            self.get_all_layers(self.net)
        
    
    def train(self , epoch , lr ):
                
        
        optimizer = optim.Adam(self.net.parameters(), lr)
        
        loss_avg = 0
        
        for e in range(epoch):
            
            
            optimizer.zero_grad()
            loss , _ , _ , _ = self.model.calculateLoss( self.BATCH_SIZE )
            loss_avg = loss_avg + float(loss.item())
            loss.backward()
            optimizer.step()
            
            if e % 50 == 49:
                
                loss = loss_avg/50
                print("Epoch {} - lr {} -  loss: {}".format(e , lr , loss ))
                loss_avg = 0

                #history_validation_de.append( validate_DE()[0] )
                
                plt.ioff()
                
                y_range = np.linspace(0, np.pi, 40, dtype=np.float)
                x_range = np.linspace(0, np.pi, 40, dtype=np.float)
                
                data = np.empty((2,1))
                
                Z = []
                for _x in x_range:
                    data[0] = _x
                    for _y in y_range:
                        data[1] = _y
                        indata = torch.Tensor(data.reshape(1,-1)).cuda()
                        Zdata = self.net(indata).detach().cpu()
                        Z.append(Zdata)
                
                
                _X, _Y = np.meshgrid(x_range, y_range, indexing='ij')
                
                Z_surface = np.reshape(Z, (x_range.shape[0], y_range.shape[0]))
                
                # plot
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_zlim([-0.2,1.2])
                ax.plot_surface( _X, _Y, Z_surface,  cmap=cm.YlOrBr_r, edgecolor='black', linewidth=0.0004, antialiased=True)
                ax.set_xlabel(' T ')
                ax.set_ylabel(' X ')
                ax.set_zlabel(' H ')
                #ax.legend(fontsize=8)
                path = "./anim/%i.png" % e
                plt.savefig(path)
                plt.close(fig)
                
                ## report detailed loss ##

                tl , dl , il , bl = self.model.calculateLoss( 2**8 )
                
                self.history_tl.append( tl )
                self.history_dl.append( dl )
                self.history_il.append( il )
                self.history_bl.append( bl )
                
                
                if self.debug == True:
                    mean = []
                    for l in self.hooks:
                        mean.append(torch.mean( self.hooks[l] ).item())
                    
                    self.history_mean_hooks.append( mean )


            
    def plot_report(self):
        
        
        fig, ax = plt.subplots(4, 1 ,constrained_layout=True)
        ax[0].plot( np.log(self.history_tl) , '-b', label='total')
        
        ax[0].set_title('total')
        fig.suptitle('Training Loss', fontsize=10)
        
        ax[1].plot( np.log(self.history_dl) )
        ax[1].set_title('diff operator')
        
        
        ax[2].plot( np.log(self.history_il) )
        ax[2].set_title('initial condition')
        
        
        ax[3].plot( np.log(self.history_bl) )
        ax[3].set_title('boundry condition')
        

    
    def hook_fn(self, m, i, o):
              self.hooks[m] = o.detach()
            
    def get_all_layers(self, net):
      for name, layer in net._modules.items():
          if isinstance(layer, nn.ModuleList):
              for n , l in layer.named_children():
                l.register_forward_hook(self.hook_fn)
          else:
              # it's a non sequential. Register a hook
              layer.register_forward_hook(self.hook_fn)
            
    
    
    
    def plot_activation_mean(self):
        
        if self.debug == False:
            print( 'error: debug is off , turn it on and train again ' )
        else:
            history = np.array(self.history_mean_hooks)
            jet= plt.get_cmap('jet')
            colors = iter(jet(np.linspace(0,1,10)))
            fig, ax = plt.subplots()
            for i in range(history.shape[1]):
                ax.plot(history[:,i], '--r', label= i , color=next(colors) )
            
            fig.suptitle('Layers activation mean value', fontsize=10)
            leg = ax.legend();

    