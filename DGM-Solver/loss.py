from libs import *


def criterion( net , x , x_initial , x_boundry_0 , x_boundry_pi , train = True):
    
    d = torch.autograd.grad(net(x), x , grad_outputs=torch.ones_like(net(x)) ,\
                            create_graph=True , retain_graph = True)
    dt = d[0][:,0].reshape(-1,1)
    dx = d[0][:,1].reshape(-1,1)
    # du/dxdx
    dxx = torch.autograd.grad(dx, x , grad_outputs=torch.ones_like(dx) ,\
                              retain_graph = True)[0][:,1].reshape(-1,1)
    
    # Domain 
    DO = ( dt - dxx )**2
    # Terminal Condition
    IC = ( (torch.sin(x_initial[:,1].reshape(-1,1) )) - net(x_initial) )**2
    BD_0  = ( net(x_boundry_0)  - torch.zeros(len(x_boundry_0), 1).cuda() )**2
    BD_pi = ( net(x_boundry_pi) - torch.zeros(len(x_boundry_pi), 1).cuda() )**2
    
    if train == True:
        return  torch.mean(DO + IC + BD_0 + BD_pi) , torch.mean( DO ) , torch.mean( IC ) , torch.mean( BD_0 + BD_pi )  
    else:
        return  DO , IC , BD_0 , BD_pi
    
    
    

def exact_solution(t,x):
    return np.sin(x)*np.exp(-1*t)