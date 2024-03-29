#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from libs import *
from train import *
from net import *
#import hook
from advection import *


net = Net( NL = 2 , NN = 20 )
net.to(torch.device("cuda:0"))  

advection = Advection(net)
#register_hook(net)
    
train = Train( net , advection , BATCH_SIZE = 2**5 , debug = True )
    
#%%


train.train( epoch = 2000 , lr = 0.001 )


#%%

train.plot_report()
train.plot_activation_mean()


#%% PLOTS

MAX_X = 1

x_range = torch.tensor(np.linspace(0, MAX_X , 100, dtype=np.float)).reshape(-1,1).cuda().float()

y = net(x_range).cpu().detach()

plt.plot(x_range.cpu(),y)
plt.plot(x_range.cpu(),advection.exact_solution(x_range.cpu()),'--')

#%% Error - diff exact and obtained solution

plt.plot( x_range.cpu() , y - advection.exact_solution(x_range.cpu()) )

#%%

torch.mean(y - advection.exact_solution(x_range.cpu()))

#%%

plt.hist( np.random.beta(2, 2, size=[10000,1]) ,100)

#%matplotlib auto