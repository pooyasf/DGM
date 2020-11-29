import torch 
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from matplotlib import cm


#%%  PRAMAETER SETUP

MAX_T = 10
MAX_X = 2
epsilon = 0.022


#%%  LOSS


def criterion( x , x_initial , x_bd_0 , x_bd_MAX_X ):
    
    d = torch.autograd.grad(net(x), x , grad_outputs=torch.ones_like(net(x)) ,\
                            create_graph=True , retain_graph = True)
    dt = d[0][:,0].reshape(-1,1)
    dx = d[0][:,1].reshape(-1,1)
    # du/dxdx
    dxx = torch.autograd.grad(dx, x , grad_outputs=torch.ones_like(dx) ,\
                              create_graph=True , retain_graph = True)[0][:,1].reshape(-1,1)
    dxxx = torch.autograd.grad(dxx, x , grad_outputs=torch.ones_like(dxx) ,\
                              retain_graph = True)[0][:,1].reshape(-1,1)
    
    
    # Domain 
    DO = ( dt + net(x)*dx + (epsilon**2)*dxxx )**2
    # Terminal Condition
    IC = ( torch.cos( np.pi*x_initial[:,1].reshape(-1,1) ) - net(x_initial) )**2
    # Boundry Condition
    BC = ( net(x_bd_0) - net(x_bd_MAX_X) )**2
    
    return  ( torch.mean( DO + IC + BC ) )


#%%

NE = 30
NL = 2

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc_input = nn.Linear(2,NE)
        torch.nn.init.xavier_uniform_(self.fc_input.weight)
        
        
        
        self.linears = nn.ModuleList([nn.Linear(NE, NE) for i in range(NL)])
        for i, l in enumerate(self.linears):    
            torch.nn.init.xavier_uniform_(l.weight)
        
        
        self.fc_output = nn.Linear(NE,1)
        torch.nn.init.xavier_uniform_(self.fc_output.weight)
 
        #self.relu = nn.ReLU()
        #self.act = torch.tanh
        self.act = torch.sigmoid
        #self.act = nn.Softsign()
        
    def forward(self, x):
        h = self.act( self.fc_input(x)  )
        
        
        for i, l in enumerate(self.linears):
            h = self.act( l(h) )
        
        out =            self.fc_output(h)
        
        return out 
    


#%%

net = Net()
net.to(torch.device("cuda:0"))  

#%% Net properties

#for name, param in net.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)




#%%

optimizer = optim.SGD(net.parameters(), lr=0.00001)
#optimizer = optim.Adam(net.parameters(), lr=0.0001)
#optimizer = optim.Rprop(net.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))


#%%

#lmbda = lambda epoch: 0.9995
#scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)


#%%


BATCH_SIZE = 2**5
loss_avg = 0



for epoch in range(20000):
    
    x = torch.cat(( torch.rand( [BATCH_SIZE,1] )*MAX_T , torch.rand( [BATCH_SIZE,1] )*MAX_X ) , dim = 1 ).cuda()
    x_initial = torch.cat(( torch.zeros(BATCH_SIZE, 1) , torch.rand( [BATCH_SIZE,1] )*MAX_X ) , dim = 1 ).cuda()

    rand_t = torch.rand( [BATCH_SIZE,1] )*MAX_T
    x_bd_0 = torch.cat(( rand_t , torch.zeros(BATCH_SIZE, 1) ) , dim = 1 ).cuda()
    x_bd_MAX_X = torch.cat(( rand_t , torch.zeros(BATCH_SIZE, 1) + MAX_X ) , dim = 1 ).cuda()


        
    x = Variable( x , requires_grad=True)
    optimizer.zero_grad()
    loss = criterion( x , x_initial , x_bd_0 , x_bd_MAX_X )
    loss_avg = loss_avg + loss.item()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        #scheduler.state_dict()['_last_lr'][0]
        print("Epoch {} - lr {} -  loss: {}".format(epoch , epoch , loss_avg / 500))
        loss_avg = 0
        #print(net.fc_input.weight.grad)
        #plot_grad_flow(net.named_parameters())
    
    #scheduler.step()
 


#%%% ERROR PLOT

'''

x_range = np.linspace(0, MAX_X, 40, dtype=np.float)
t_range = np.linspace(0, MAX_T, 40, dtype=np.float)

data = np.empty((2,1))

Z = []
for _t in t_range:
    data[0] = _t
    for _x in x_range:
        data[1] = _x
        indata = torch.Tensor(data.reshape(1,-1)).cuda()
        Zdata = abs(net(indata).detach().cpu() - np.sin(_x)*np.exp(-1*_t) )
        Z.append(Zdata)


_T, _X = np.meshgrid(t_range, x_range, indexing='ij')

Z_surface = np.reshape(Z, (t_range.shape[0], x_range.shape[0]))

# plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface( _T, _X, Z_surface,  cmap=cm.YlOrBr_r, edgecolor='gray', linewidth=0.004, antialiased=False)
plt.show()

'''


#%%% SURFACE PLOT


y_range = np.linspace(0, MAX_X, 40, dtype=np.float)
x_range = np.linspace(0, MAX_T, 40, dtype=np.float)

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
plt.show()


#%%

#for name, param in net.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)


#%%



net( torch.tensor([0,0]).cuda().float())
net( torch.tensor([0,MAX_X]).cuda().float())
