from libs import *

class Advection():
    
    def __init__(self , net ):
        
        self.net = net

    def sample(self , xs = 0 , xe = 1 , size = 2**8 ):
         
         #uniform
         x = (torch.rand( [size,1] )*xe  ).cuda()
            
         x_initial = torch.zeros(1, 1).cuda() 
         
         return x , x_initial
    
    
    def criterion(self , x  , x_initial):
        
        d = torch.autograd.grad(self.net(x), x , grad_outputs=torch.ones_like(self.net(x)) ,\
                                create_graph=True)
        
        dx = d[0].reshape(-1,1)
        
        # Domain 
        DO = ( dx - 2*np.pi*torch.cos(2*np.pi*x)*torch.cos(4*np.pi*x) + 4*np.pi*torch.sin(4*np.pi*x)*torch.sin(2*np.pi*x) )**2
        IC  = ( self.net(x_initial)  - torch.ones(1, 1).cuda() )**2
        
        return  DO , IC


    def calculateLoss(self , size = 2**8 , train = True):
        
        x , x_boundry_0 = self.sample(size)
        x = Variable( x , requires_grad = True )
        DO , IC  = self.criterion( x , x_boundry_0 )
        
        if train == True:
            return  torch.mean( DO ) + IC , torch.mean( DO ) ,  torch.mean( IC )  
        else:
            return  DO , IC 
        
    
    def exact_solution( self , x ):
        return np.sin(2*np.pi*x)*np.cos(4*np.pi*x)+1