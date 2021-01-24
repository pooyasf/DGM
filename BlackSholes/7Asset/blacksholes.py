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
        return torch.max( (x[:,1].reshape(-1,1)*x[:,2].reshape(-1,1)*x[:,3].reshape(-1,1)*x[:,4].reshape(-1,1)*x[:,5].reshape(-1,1)*x[:,6].reshape(-1,1)*x[:,7].reshape(-1,1))**(1/3) - self.K , torch.zeros([len(x),1]).cuda() ) 

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
        x = torch.cat(( torch.rand( [size,1] )*self.T , torch.rand( [size,1] )*self.MAX_X \
                                                          , torch.rand( [size,1] )*self.MAX_X \
                                                              , torch.rand( [size,1] )*self.MAX_X \
                                                                  , torch.rand( [size,1] )*self.MAX_X \
                                                                      , torch.rand( [size,1] )*self.MAX_X \
                                                                          , torch.rand( [size,1] )*self.MAX_X \
                                                                              , torch.rand( [size,1] )*self.MAX_X  \
                                                                                  ) , dim = 1 ).cuda()
        compare = self.net(x) - self.g(x)
        mask = compare > 0
        ## repeat for each 7 assets + 1 time dimension
        mask = mask.repeat(1,8)
        x = x[mask].reshape(-1,8)
        ###
        
        ### Terminal
        x_terminal = torch.cat(( torch.zeros(size, 1) + self.T , torch.rand( [size,1] )*self.MAX_X \
                                                                    , torch.rand( [size,1] )*self.MAX_X \
                                                                        , torch.rand( [size,1] )*self.MAX_X \
                                                                            , torch.rand( [size,1] )*self.MAX_X \
                                                                                , torch.rand( [size,1] )*self.MAX_X \
                                                                                    , torch.rand( [size,1] )*self.MAX_X \
                                                                                        , torch.rand( [size,1] )*self.MAX_X \
                                                                            ) , dim = 1 ).cuda()
        ###
        
        ### under Free Boundry
        x_boundry = torch.cat(( torch.rand( [size,1] )*self.T , torch.rand( [size,1] )*self.MAX_X \
                                                                   , torch.rand( [size,1] )*self.MAX_X \
                                                                       , torch.rand( [size,1] )*self.MAX_X \
                                                                           , torch.rand( [size,1] )*self.MAX_X \
                                                                               , torch.rand( [size,1] )*self.MAX_X \
                                                                                   , torch.rand( [size,1] )*self.MAX_X \
                                                                                       , torch.rand( [size,1] )*self.MAX_X \
                                                                           ) , dim = 1 ).cuda()
        compare = self.net(x_boundry) - self.g(x_boundry)
        mask = compare < 0
        ## repeat for each d (dimension)
        mask = mask.repeat(1,8)
        x_boundry = x_boundry[mask].reshape(-1,8)
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
        dx4 = d[0][:,4].reshape(-1,1)
        dx5 = d[0][:,5].reshape(-1,1)
        dx6 = d[0][:,6].reshape(-1,1)
        dx7 = d[0][:,7].reshape(-1,1)
        
        # du/dxdx
        
        dx1x = torch.autograd.grad(dx1, x , grad_outputs=torch.ones_like(dx1) ,\
                                  create_graph = True)
        dx1x1 = d[0][:,1].reshape(-1,1)
        dx1x2 = d[0][:,2].reshape(-1,1)
        dx1x3 = d[0][:,3].reshape(-1,1)
        dx1x4 = d[0][:,4].reshape(-1,1)
        dx1x5 = d[0][:,5].reshape(-1,1)
        dx1x6 = d[0][:,6].reshape(-1,1)
        dx1x7 = d[0][:,7].reshape(-1,1)
        
        
        dx2x = torch.autograd.grad(dx2, x , grad_outputs=torch.ones_like(dx2) ,\
                                  create_graph = True)
        dx2x1 = d[0][:,1].reshape(-1,1)
        dx2x2 = d[0][:,2].reshape(-1,1)
        dx2x3 = d[0][:,3].reshape(-1,1)
        dx2x4 = d[0][:,4].reshape(-1,1)
        dx2x5 = d[0][:,5].reshape(-1,1)
        dx2x6 = d[0][:,6].reshape(-1,1)
        dx2x7 = d[0][:,7].reshape(-1,1)

        dx3x = torch.autograd.grad(dx3, x , grad_outputs=torch.ones_like(dx3) ,\
                                  create_graph = True)
        dx3x1 = d[0][:,1].reshape(-1,1)
        dx3x2 = d[0][:,2].reshape(-1,1)
        dx3x3 = d[0][:,3].reshape(-1,1)
        dx3x4 = d[0][:,4].reshape(-1,1)
        dx3x5 = d[0][:,5].reshape(-1,1)
        dx3x6 = d[0][:,6].reshape(-1,1)
        dx3x7 = d[0][:,7].reshape(-1,1)
        
        dx4x = torch.autograd.grad(dx4, x , grad_outputs=torch.ones_like(dx4) ,\
                                  create_graph = True)
        dx4x1 = d[0][:,1].reshape(-1,1)
        dx4x2 = d[0][:,2].reshape(-1,1)
        dx4x3 = d[0][:,3].reshape(-1,1)
        dx4x4 = d[0][:,4].reshape(-1,1)
        dx4x5 = d[0][:,5].reshape(-1,1)
        dx4x6 = d[0][:,6].reshape(-1,1)
        dx4x7 = d[0][:,7].reshape(-1,1)
        
        dx5x = torch.autograd.grad(dx5, x , grad_outputs=torch.ones_like(dx5) ,\
                                  create_graph = True)
        dx5x1 = d[0][:,1].reshape(-1,1)
        dx5x2 = d[0][:,2].reshape(-1,1)
        dx5x3 = d[0][:,3].reshape(-1,1)
        dx5x4 = d[0][:,4].reshape(-1,1)
        dx5x5 = d[0][:,5].reshape(-1,1)
        dx5x6 = d[0][:,6].reshape(-1,1)
        dx5x7 = d[0][:,7].reshape(-1,1)
        
        
        dx6x = torch.autograd.grad(dx6, x , grad_outputs=torch.ones_like(dx6) ,\
                                  create_graph = True)
        dx6x1 = d[0][:,1].reshape(-1,1)
        dx6x2 = d[0][:,2].reshape(-1,1)
        dx6x3 = d[0][:,3].reshape(-1,1)
        dx6x4 = d[0][:,4].reshape(-1,1)
        dx6x5 = d[0][:,5].reshape(-1,1)
        dx6x6 = d[0][:,6].reshape(-1,1)
        dx6x7 = d[0][:,7].reshape(-1,1)
        
        dx7x = torch.autograd.grad(dx7, x , grad_outputs=torch.ones_like(dx7) ,\
                                  create_graph = True)
        dx7x1 = d[0][:,1].reshape(-1,1)
        dx7x2 = d[0][:,2].reshape(-1,1)
        dx7x3 = d[0][:,3].reshape(-1,1)
        dx7x4 = d[0][:,4].reshape(-1,1)
        dx7x5 = d[0][:,5].reshape(-1,1)
        dx7x6 = d[0][:,6].reshape(-1,1)
        dx7x7 = d[0][:,7].reshape(-1,1)
        
        
        
        if len(x) == 0:
            print('zero batch size for domain!')
            
        DO = ( dt + self.mu(x[:,1])*( dx1 ) + self.mu(x[:,2])*( dx2 ) + self.mu(x[:,3])*( dx3 ) + self.mu(x[:,3])*( dx4 ) + self.mu(x[:,3])*( dx5 ) + self.mu(x[:,3])*( dx6 ) + self.mu(x[:,3])*( dx7 ) \
                  + 0.5*(    1*(self.sigma(x[:,1])*self.sigma(x[:,1]))*dx1x1  \
                                + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,2]))*dx1x2  \
                                    + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,3]))*dx1x3  \
                                        + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,4]))*dx1x4  \
                                            + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,5]))*dx1x5  \
                                                + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,5]))*dx1x6  \
                                                    + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,5]))*dx1x7  \
                            + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,1]))*dx2x1  \
                                + 1*(self.sigma(x[:,2])*self.sigma(x[:,2]))*dx2x2  \
                                    + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,3]))*dx2x3  \
                                        + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,4]))*dx2x4  \
                                            + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,5]))*dx2x5  \
                                                + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,6]))*dx2x6  \
                                                    + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,7]))*dx2x7  \
                            + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,1]))*dx3x1  \
                                + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,2]))*dx3x2  \
                                    + 1*((self.sigma(x[:,3])*self.sigma(x[:,3]))*dx3x3 ) \
                                         + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,4]))*dx3x4  \
                                             + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,5]))*dx3x5  \
                                                 + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,6]))*dx3x6  \
                                                     + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,7]))*dx3x7  \
                            + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,1]))*dx4x1  \
                                + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,2]))*dx4x2  \
                                    + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,3]))*dx4x3  \
                                         + 1*(self.sigma(x[:,4])*self.sigma(x[:,4]))*dx4x4  \
                                             + self.RU*(self.sigma(x[:4])*self.sigma(x[:,5]))*dx4x5  \
                                                 + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,6]))*dx4x6  \
                                                     + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,7]))*dx4x7  \
                            + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,1]))*dx5x1  \
                                + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,2]))*dx5x2  \
                                    + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,3]))*dx5x3  \
                                         + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,4]))*dx5x4  \
                                             + 1*(self.sigma(x[:5])*self.sigma(x[:,5]))*dx5x5  \
                                                 + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,6]))*dx5x6  \
                                                     + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,7]))*dx5x7  \
                            + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,1]))*dx6x1  \
                                + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,2]))*dx6x2  \
                                    + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,3]))*dx6x3  \
                                         + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,4]))*dx6x4  \
                                             + self.RU*(self.sigma(x[:6])*self.sigma(x[:,6]))*dx6x5  \
                                                 + 1*(self.sigma(x[:,6])*self.sigma(x[:,6]))*dx6x6  \
                                                     + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,7]))*dx6x7  \
                            + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,1]))*dx7x1  \
                                + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,2]))*dx7x2  \
                                    + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,3]))*dx7x3  \
                                         + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,4]))*dx7x4  \
                                             + self.RU*(self.sigma(x[:7])*self.sigma(x[:,6]))*dx7x5  \
                                                 + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,6]))*dx7x6  \
                                                     + 1*(self.sigma(x[:,7])*self.sigma(x[:,7]))*dx7x7  \
                                                       ) - self.R*self.net(x) )**2
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