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
            loss , loss_DO , loss_Initial  = self.model.calculateLoss( self.BATCH_SIZE )
            loss_avg = loss_avg + float(loss.item())
            loss.backward()
            optimizer.step()          
            
            # accessing gradient of weights during learning
            # to do : collect and plot histogram!
            

            if e % 50 == 49:
                
                loss = loss_avg/50
                print("Epoch {} - lr {} -  loss: {}".format(e , lr , loss ))
                loss_avg = 0
                
                
                # save for animation!
                
                plt.ioff()
                MAX_X = 1

                x_range = torch.tensor(np.linspace(0, MAX_X , 100, dtype=np.float)).reshape(-1,1).cuda().float()

                y = self.net(x_range).cpu().detach()
                fig, ax = plt.subplots()
                ax.set_ylim([-0.5,2.5])
                ax.plot(x_range.cpu(),y,label='Neural Net')
                ax.plot(x_range.cpu(),self.model.exact_solution(x_range.cpu()),'--',color='lightgray',label='Exact')
                ax.legend(fontsize=8)
                path = "./anim/%i.png" % e
                plt.savefig(path)
                plt.close(fig)
                
                #print(torch.mean(self.net.fc_input.weight.grad))
                #print(torch.mean(self.net.linears[0].weight.grad))
                #print(torch.mean(self.net.linears[1].weight.grad))
                #print(torch.mean(self.net.fc_output.weight.grad))
                #history_validation_de.append( validate_DE()[0] )
                
                ## report detailed loss ##

                tl , dl , il = self.model.calculateLoss( 2**10 )
                
                self.history_tl.append( float(tl) )
                self.history_dl.append( float(dl) )
                self.history_il.append( float(il) )
                
                
                if self.debug == True:
                    mean = []
                    for l in self.hooks:
                        mean.append(torch.mean( self.hooks[l] ).item())
                    
                    self.history_mean_hooks.append( mean )


            
    def plot_report(self):
        
        
        fig, ax = plt.subplots(3, 1 ,constrained_layout=True)
        ax[0].plot( np.log(self.history_tl) , '-b', label='total')
        
        ax[0].set_title('total')
        fig.suptitle('Training Loss', fontsize=10)
        
        ax[1].plot( np.log(self.history_dl) )
        ax[1].set_title('diff operator')
        
        
        ax[2].plot( np.log(self.history_il) )
        ax[2].set_title('initial condition')
        
        
        #ax[3].plot( np.log(self.history_bl) )
        #ax[3].set_title('boundry condition')
        

    
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

    