from libs import *

class Heat():
    
    def __init__(self , net):
        
        ## for accept reject purpose!
        ## free boundry problems
        self.net = net

    def sample(self , ts = 0 , te = np.pi , xs = 0 , xe = np.pi , size = 2**8 ):
         
         x = torch.cat(( torch.rand( [size,1] )*te  , torch.rand( [size,1] )*xe  ) , dim = 1 ).cuda()
         x_initial = torch.cat(( torch.zeros(size, 1)  , torch.rand( [size,1] )*xe  ) , dim = 1 ).cuda()
            
         x_boundry_start = torch.cat(( torch.rand( [size,1] )*te , torch.zeros(size, 1)  ) , dim = 1 ).cuda()
         x_boundry_end = torch.cat(( torch.rand( [size,1] )*te  , torch.zeros(size, 1) + xe  ) , dim = 1 ).cuda()
         
         return x , x_initial , x_boundry_start , x_boundry_end
    
    
    def criterion(self , x , x_initial , x_boundry_0 , x_boundry_pi):
        
        d = torch.autograd.grad(self.net(x), x , grad_outputs=torch.ones_like(self.net(x)) ,\
                                create_graph=True )
        dt = d[0][:,0].reshape(-1,1)
        dx = d[0][:,1].reshape(-1,1)
        # du/dxdx
        dxx = torch.autograd.grad(dx, x , grad_outputs=torch.ones_like(dx) ,\
                                  create_graph = True)[0][:,1].reshape(-1,1)
        
        # Domain 
        DO = ( dt - dxx )**2
        # Terminal Condition
        IC = ( (torch.sin(x_initial[:,1].reshape(-1,1) )) - self.net(x_initial) )**2
        BD_0  = ( self.net(x_boundry_0)  - torch.zeros(len(x_boundry_0), 1).cuda() )**2
        BD_pi = ( self.net(x_boundry_pi) - torch.zeros(len(x_boundry_pi), 1).cuda() )**2
        
        return  DO , IC , BD_0 , BD_pi


    def calculateLoss(self , size = 2**8 , train = True):
        
        x , x_initial , x_boundry_0 , x_boundry_pi = self.sample(size)
        x = Variable( x , requires_grad=True)
        DO , IC , BD_0  , BD_pi = self.criterion( x , x_initial , x_boundry_0 , x_boundry_pi )
        
        if train == True:
            return  torch.mean(DO + IC + BD_0 + BD_pi) , torch.mean( DO ) , torch.mean( IC ) , torch.mean( BD_0 + BD_pi )  
        else:
            return  DO , IC , BD_0 , BD_pi
        
    
    def exact_solution(self , t , x ):
        return np.sin(x)*np.exp(-1*t)