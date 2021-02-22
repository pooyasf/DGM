from libs import *
from train import *
from net import *
#import hook
from blacksholes import *


net = Net( NL = 2 , NN = 20 )
net.to(torch.device("cuda:0"))  

## providing sampler with net so it can accept/reject based on net and other criterions
bsequation = BlackSholes(net)
#register_hook(net)
    
train = Train( net , bsequation , BATCH_SIZE = 2**8 , debug = True )

#%%

train.train( epoch = 3000 , lr = 0.000005 )


#%%

train.plot_report()
train.plot_activation_mean()


#%%% SURFACE PLOT


y_range = np.linspace(0, 2, 40, dtype=np.float)
x_range = np.linspace(0, 2, 40, dtype=np.float)

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
ax.set_xlabel('Time')
ax.set_ylabel('Asset  Price')
ax.set_zlabel('Option Price')
plt.show()

#%%%

print( 'Value at 0' , float(net( torch.tensor( [ 0.0 , 1. ] ).cuda() )) )

print( 'Value at expire' , float(net( torch.tensor( [ 2. , 1. ] ).cuda() )) )

#%% PLOTS

# MAX_T = np.pi
# MAX_X = np.pi

# t_range = np.linspace(0, MAX_T , 100, dtype=np.float)
# x_range = np.linspace(0, MAX_X , 100, dtype=np.float)

# _T, _X = np.meshgrid(t_range, x_range, indexing='ij')


# x = torch.tensor( np.concatenate( (_T.reshape(-1,1) , _X.reshape(-1,1)) , axis = 1 ) )
# x = Variable(x , requires_grad = True).cuda().float()
# tl , dl , il , bl = heatequation.criterion( x , x , x , x )
# Z_surface = torch.reshape(tl, (t_range.shape[0], x_range.shape[0]))

# fig, ax = plt.subplots()
# CS = ax.contour(_T , _X , Z_surface.cpu().detach() , levels = 10)
# ax.clabel(CS, inline=1, fontsize=8)
# CB = fig.colorbar(CS, shrink=0.8, extend='both')
# ax.set_title(' differential operators error ')
# ax.set_xlabel(' t ', fontsize=10)
# ax.set_ylabel(' x ', fontsize=10)

# z = net(x)
# Z_surface = torch.reshape(z, (t_range.shape[0], x_range.shape[0]))

# fig, ax = plt.subplots()
# CS = ax.contour(_T , _X , Z_surface.cpu().detach() , levels = 10)
# ax.clabel(CS, inline=1, fontsize=8)
# CB = fig.colorbar(CS, shrink=0.8, extend='both')
# ax.set_title(' solution ')
# ax.set_xlabel(' t ', fontsize=10)
# ax.set_ylabel(' x ', fontsize=10)


# z = net(x).cpu().detach() - heatequation.exact_solution( _T.reshape(-1,1) , _X.reshape(-1,1) )
# mse_error = torch.mean( z**2 )
# Z_surface = torch.reshape(z, (t_range.shape[0], x_range.shape[0]))

# fig, ax = plt.subplots()
# CS = ax.contour(_T , _X , Z_surface.cpu().detach() , levels = 10)
# ax.clabel(CS, inline=1, fontsize=8)
# CB = fig.colorbar(CS, shrink=0.8, extend='both')
# ax.set_title(' net(x) - exact solution \n MSE: %f ' %mse_error.item())
# ax.set_xlabel(' t ', fontsize=10)
# ax.set_ylabel(' x ', fontsize=10)

# #%%

# x , x_initial , x_boundry_0 , x_boundry_pi = heatequation.sample( ts = 2 , te = 3 , xs = 2 , xe = 3 , size = 2**12 )

# out = net(x)

# train.hooks

# #%%

# jet= plt.get_cmap('jet')
# colors = iter(jet(np.linspace(0,1,10)))
# fig, ax = plt.subplots()
            
# for i in train.hooks:
#     #print(train.hooks[i].reshape(1,-1))
#     ax.hist( train.hooks[i].reshape(1,-1).cpu().detach() , label= i , color=next(colors) ,\
#             bins = np.linspace( -4 , 4 , 20) , histtype = 'step' , log = True , density = True )
# fig.suptitle('Layers activation hist at end of training', fontsize=10)
# leg = ax.legend();



#%% Running for different archs


torch.save(net.state_dict(), './model')

#%%

net = TheModelClass(*args, **kwargs)
net.load_state_dict(torch.load('./model'))
net.eval()





