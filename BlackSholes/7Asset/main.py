from libs import *
from train import *
from net import *
#import hook
from blacksholes import *

#integration

net = Net( NL = 4 , NN = 16 )
#net = DGMNet()
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
bsequation = BlackSholes(net)
#register_hook(net)
    
train = Train( net , bsequation , BATCH_SIZE = 2**10 , debug = True )

#%%   

train.train( epoch = 1000 , lr = 0.001 )
train.train( epoch = 1000 , lr = 0.0001 )
train.train( epoch = 1000 , lr = 0.001 )
train.train( epoch = 1000 , lr = 0.0001 )

#%%

train.plot_report()
train.plot_activation_mean()



#%%%

option_xt = torch.tensor( [[ 0. , 1. , 1. , 1. , 1. , 1. , 1. , 1.  ]] ).cuda()

print( 'Option Value' , net.train( option_xt ).cuda() )


#%% save


torch.save(net.state_dict(), './model7AssetsMLP')

#%%

net = TheModelClass(*args, **kwargs)
net.load_state_dict(torch.load('./modelmodel7Assets'))
net.eval()


#%%
x = torch.tensor([[0,1,1,1,1,1,1,1]]).cuda().float()
x = Variable( x , requires_grad=True)


a , b , c = bsequation.criterion( x, torch.tensor([[0,0,0,0,0,0,0,0]]).cuda().float() , torch.tensor([[0,0,0,0,0,0,0,0]]).cuda().float())

x , _ , _ = bsequation.sample()
plt.hist(x.cpu()[:,0])

#%%



