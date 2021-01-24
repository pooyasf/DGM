from libs import *

class BlackSholes():
    
    def __init__(self , net):
        
        
        self.C = 0.02
        self.R = 0       # Interest Rate (Yearly)
        self.SIGMA = 0.25   # Volatility (Yearly)
        self.RU = 0.75        # stock corrolation
        self.K = 1.0       # Strike Price 
        self.T = 2.0       # Maturation time (in YEAR)
        self.MAX_X = 2.0       # MAX price
        
        ## for accept reject purpose!
        ## free boundry problems
        self.net = net

    def g(self , x):
        return torch.max( (x[:,1].reshape(-1,1)*x[:,2].reshape(-1,1)*x[:,3].reshape(-1,1))**(1/3) - self.K , torch.zeros([len(x),1]).cuda() ) 

    def mu(self, x):
        ## should test it! output dimension is important !
        return (self.R-self.C)*x.reshape(-1,1)

    def sigma(self , x):
        return self.SIGMA*x.reshape(-1,1)

    def sample(self , size = 2**8 ):
        '''
        Sampling function



        '''
         
        ### Domain (above boundry)
        x = torch.cat(( torch.rand( [size,1] )*self.T , torch.rand( [size,1] )*self.MAX_X , torch.rand( [size,1] )*self.MAX_X , torch.rand( [size,1] )*self.MAX_X ) , dim = 1 ).cuda()
        compare = self.net(x) - self.g(x)
        mask = compare > 0
        ## repeat for each d (dimension)
        mask = mask.repeat(1,4)
        x = x[mask].reshape(-1,4)
        ###
        
        ### Terminal
        x_terminal = torch.cat(( torch.zeros(size, 1) + self.T , torch.rand( [size,1] )*self.MAX_X , torch.rand( [size,1] )*self.MAX_X , torch.rand( [size,1] )*self.MAX_X ) , dim = 1 ).cuda()
        ###
        
        ### under Free Boundry
        x_boundry = torch.cat(( torch.rand( [size,1] )*self.T , torch.rand( [size,1] )*self.MAX_X , torch.rand( [size,1] )*self.MAX_X , torch.rand( [size,1] )*self.MAX_X ) , dim = 1 ).cuda()
        compare = self.net(x_boundry) - self.g(x_boundry)
        mask = compare < 0
        ## repeat for each d (dimension)
        mask = mask.repeat(1,4)
        x_boundry = x_boundry[mask].reshape(-1,4)
        ###
         
        return x , x_terminal , x_boundry
    
    
    def criterion(self , x , x_terminal , x_boundry):
        '''
        Loss function that helps network find solution to equation

        '''
        
        d = torch.autograd.grad(self.net(x), x , grad_outputs=torch.ones_like(self.net(x)) ,\
                            create_graph=True )
        
        dt  = d[0][:,0].reshape(-1,1)
        dx1 = d[0][:,1].reshape(-1,1)
        dx2 = d[0][:,2].reshape(-1,1)
        dx3 = d[0][:,3].reshape(-1,1)
        
        # du/dxdx
        
        dx1x1 = torch.autograd.grad(dx1, x , grad_outputs=torch.ones_like(dx1) ,\
                                  create_graph = True)[0][:,1].reshape(-1,1)
        dx1x2 = torch.autograd.grad(dx1, x , grad_outputs=torch.ones_like(dx1) ,\
                                  create_graph = True)[0][:,2].reshape(-1,1)
        dx1x3 = torch.autograd.grad(dx1, x , grad_outputs=torch.ones_like(dx1) ,\
                                  create_graph = True)[0][:,3].reshape(-1,1)
            
        dx2x1 = torch.autograd.grad(dx2, x , grad_outputs=torch.ones_like(dx2) ,\
                                  create_graph = True)[0][:,1].reshape(-1,1)
        dx2x2 = torch.autograd.grad(dx2, x , grad_outputs=torch.ones_like(dx2) ,\
                                  create_graph = True)[0][:,2].reshape(-1,1)
        dx2x3 = torch.autograd.grad(dx2, x , grad_outputs=torch.ones_like(dx2) ,\
                                  create_graph = True)[0][:,3].reshape(-1,1)
            
        dx3x1 = torch.autograd.grad(dx3, x , grad_outputs=torch.ones_like(dx3) ,\
                                  create_graph = True)[0][:,1].reshape(-1,1)
        dx3x2 = torch.autograd.grad(dx3, x , grad_outputs=torch.ones_like(dx3) ,\
                                  create_graph = True)[0][:,2].reshape(-1,1)
        dx3x3 = torch.autograd.grad(dx3, x , grad_outputs=torch.ones_like(dx3) ,\
                                  create_graph = True)[0][:,3].reshape(-1,1)
        
        if len(x) == 0:
            print('zero batch size for domain!')
        DO = ( dt + self.mu(x[:,1])*( dx1 ) + self.mu(x[:,2])*( dx2 ) + self.mu(x[:,3])*( dx3 ) \
                  + 0.5*(  (self.sigma(x[:,1])*self.sigma(x[:,1]))*dx1x1  \
                            + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,2]))*dx1x2  \
                            + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,3]))*dx1x3  \
                            + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,1]))*dx2x1  \
                            + (self.sigma(x[:,2])*self.sigma(x[:,2]))*dx2x2  \
                            + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,3]))*dx2x3  \
                            + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,1]))*dx3x1  \
                            + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,2]))*dx3x2  \
                            + (self.sigma(x[:,3])*self.sigma(x[:,3]))*dx3x3 ) \
              - self.R*self.net(x) )**2
        # Domain 
        #DO = (dt + 0.5*self.RU*(self.SIGMA**2)*(x[:,1].reshape(-1,1)**2)*dxx - self.R*self.net(x) + (self.R-self.C)*x[:,1].reshape(-1,1)*dx)**2
        
        # Terminal Condition
        TC = ( self.g(x_terminal) - self.net(x_terminal))**2 
        
        # Boundry Condition
        # len() is safe here , because it just shows batch number 
        if( len(x_boundry) != 0):
            BC = torch.max( self.g(x_boundry) - self.net(x_boundry) , torch.zeros([len(x_boundry),1]).cuda() )**2 
        else:
            print('zero batch size for outside domain!')
            BC = torch.tensor(0).cuda().float()
        
        return  DO , TC , BC


    def calculateLoss(self , size = 2**8 , train = True):
        '''
        
        Helper function that Sample and Calculate loss


        '''        
        x , x_terminal , x_boundry = self.sample(size)
        x = Variable( x , requires_grad=True)
        DO , TC , BC = self.criterion( x , x_terminal , x_boundry )
        
        if train == True:
            return  (torch.mean(DO) + torch.mean(TC) + torch.mean(BC)) , torch.mean( DO ) , torch.mean( TC ) , torch.mean( BC )  
        else:
            return  DO , TC , BC