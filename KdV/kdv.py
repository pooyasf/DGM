from libs import *

class KDV():
    
    def __init__(self , net):
        
        self.net = net
        self.MAX_T = 1.145
        self.MAX_X = 2
        self.epsilon = 0.022

    def sample(self , size ):
         
         #seed = torch.rand(1)*1000
         #x = torch.tensor(chaospy.create_sobol_samples(size,8,seed).reshape(size,8)*self.MAX_X).cuda().float()
        
         x = torch.cat(( torch.rand( [size,1] )*(self.MAX_T) - self.MAX_T/2 , torch.rand( [size,1] )*self.MAX_X - self.MAX_X/2 ) , dim = 1 ).cuda()
         #r_normal = torch.normal(self.MAX_T/2, 1, size=(size*2, 1))
         #ix_upper = (r_normal < 1.145/2)&(r_normal > -1.145/2)
         #clipped_normal = r_normal[ix_upper].reshape(-1,1)
         
         #x = torch.cat(( clipped_normal , torch.rand( [len(clipped_normal),1] )*self.MAX_X - self.MAX_X/2 ) , dim = 1 ).cuda()
         
         x_initial = torch.cat(( torch.zeros(size, 1) - self.MAX_T/2 , torch.rand( [size,1] )*self.MAX_X - self.MAX_X/2 ) , dim = 1 ).cuda()

         rand_t = torch.rand( [size,1] )*self.MAX_T - self.MAX_T/2
         x_boundry_start = torch.cat(( rand_t , torch.zeros(size, 1) - self.MAX_X/2 ) , dim = 1 ).cuda()
         x_boundry_end = torch.cat(( rand_t , torch.zeros(size, 1)   + self.MAX_X/2 ) , dim = 1 ).cuda()
         
         return x , x_initial , x_boundry_start , x_boundry_end
    
    def sech(self , x ):
    
        return (torch.exp(x) - torch.exp(-x))/2 
    
    def criterion(self , x , x_initial , x_boundry_start , x_boundry_end):
        
        d = torch.autograd.grad(self.net(x), x , grad_outputs=torch.ones_like(self.net(x)) ,\
                            create_graph=True )
        dt = d[0][:,0].reshape(-1,1)
        dx = d[0][:,1].reshape(-1,1)
        # du/dxdx
        dxx = torch.autograd.grad(dx, x , grad_outputs=torch.ones_like(dx) ,\
                                  create_graph=True )[0][:,1].reshape(-1,1)
        dxxx = torch.autograd.grad(dxx, x , grad_outputs=torch.ones_like(dxx) ,\
                                  create_graph=True )[0][:,1].reshape(-1,1)
        
        # Domain 
        #DO = torch.tensor(0).float().cuda() 
        DO = ( dt + self.net(x)*dx + (self.epsilon**2)*dxxx )**2
        # Terminal Condition
        IC = ( torch.cos( np.pi*(x_initial[:,1].reshape(-1,1) + torch.ones(len(x_initial[:,1]), 1).cuda() ) ) - self.net(x_initial) )**2
        #  one soliton initial condition ,, check wolfram demonstration project
        #IC = ( (1/torch.cosh( (x_initial[:,1].reshape(-1,1) + torch.ones(len(x_initial[:,1]), 1).cuda()   ) ))**2 - self.net(x_initial) )**2
        #IC = ( self.sech( x_initial[:,1].reshape(-1,1) )**2  -  self.net(x_initial)  )**2
        # Boundry Condition
        BC = ( self.net(x_boundry_start) - self.net(x_boundry_end) )**2
        
        return  DO , IC , BC


    def calculateLoss(self , size , train = True):
        
        x , x_initial , x_boundry_start , x_boundry_end = self.sample(size)
        x = Variable( x , requires_grad=True)
        DO , IC , BC = self.criterion( x , x_initial , x_boundry_start , x_boundry_end )
        
        if train == True:
            return  torch.mean(DO) + torch.mean(IC + BC) , torch.mean( DO ) , torch.mean( IC ) , torch.mean( BC )  
        else:
            return  DO , IC , BC