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
        return torch.max( (x[:,1].reshape(-1,1)*x[:,2].reshape(-1,1)*x[:,3].reshape(-1,1)*x[:,4].reshape(-1,1)*x[:,5].reshape(-1,1)*x[:,6].reshape(-1,1)*x[:,7].reshape(-1,1))**(1/7) - self.K , torch.zeros([len(x),1]).cuda() ) 

    def mu(self, x):
        ## should test it! output dimension is important !
        return (self.R-self.C)*x.reshape(-1,1)

    def sigma(self , x):
        return self.SIGMA*x.reshape(-1,1)
        

    def sample(self , size = 2**8 ):
        '''
        Sampling function

        '''
        #seed = torch.rand(1)*1000
        #x = torch.tensor(chaospy.create_sobol_samples(size,8,seed).reshape(size,8)*self.MAX_X).cuda().float()


        ### Domain (above boundry)
        x = torch.cat(( torch.rand( [size,1] )*self.T , torch.rand( [size,1] )*self.MAX_X \
                                                          , torch.rand( [size,1] )*self.MAX_X \
                                                              , torch.rand( [size,1] )*self.MAX_X \
                                                                  , torch.rand( [size,1] )*self.MAX_X \
                                                                      , torch.rand( [size,1] )*self.MAX_X \
                                                                          , torch.rand( [size,1] )*self.MAX_X \
                                                                              , torch.rand( [size,1] )*self.MAX_X  \
                                                                                  ) , dim = 1 ).cuda()
        # x_initial_point = torch.tensor([[ 0. , 1. , 1., 1., 1., 1., 1., 1. ]]).cuda()
        # x = torch.cat( ( x , x_initial_point  ) , dim = 0 ).cuda()
        compare = self.net(x) - self.g(x)
        mask = compare > 0
        ## repeat for each 7 assets + 1 time dimension
        mask = mask.repeat(1,8)
        x = x[mask].reshape(-1,8)
        ###
        
        ### Terminal
        #x_terminal = torch.tensor(chaospy.create_sobol_samples(size,7,seed).reshape(size,7)*self.MAX_X).float()
        #x_terminal = torch.cat( (torch.zeros(size, 1) + self.T , x_terminal) , dim = 1 ).cuda()
        
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
        #x_boundry = torch.tensor(chaospy.create_sobol_samples(size,8,seed).reshape(size,8)*self.MAX_X).cuda().float()
        
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
        dx1x1 = dx1x[0][:,1].reshape(-1,1)
        dx1x2 = dx1x[0][:,2].reshape(-1,1)
        dx1x3 = dx1x[0][:,3].reshape(-1,1)
        dx1x4 = dx1x[0][:,4].reshape(-1,1)
        dx1x5 = dx1x[0][:,5].reshape(-1,1)
        dx1x6 = dx1x[0][:,6].reshape(-1,1)
        dx1x7 = dx1x[0][:,7].reshape(-1,1)
        
        
        dx2x = torch.autograd.grad(dx2, x , grad_outputs=torch.ones_like(dx2) ,\
                                  create_graph = True)
        dx2x1 = dx2x[0][:,1].reshape(-1,1)
        dx2x2 = dx2x[0][:,2].reshape(-1,1)
        dx2x3 = dx2x[0][:,3].reshape(-1,1)
        dx2x4 = dx2x[0][:,4].reshape(-1,1)
        dx2x5 = dx2x[0][:,5].reshape(-1,1)
        dx2x6 = dx2x[0][:,6].reshape(-1,1)
        dx2x7 = dx2x[0][:,7].reshape(-1,1)

        dx3x = torch.autograd.grad(dx3, x , grad_outputs=torch.ones_like(dx3) ,\
                                  create_graph = True)
        dx3x1 = dx3x[0][:,1].reshape(-1,1)
        dx3x2 = dx3x[0][:,2].reshape(-1,1)
        dx3x3 = dx3x[0][:,3].reshape(-1,1)
        dx3x4 = dx3x[0][:,4].reshape(-1,1)
        dx3x5 = dx3x[0][:,5].reshape(-1,1)
        dx3x6 = dx3x[0][:,6].reshape(-1,1)
        dx3x7 = dx3x[0][:,7].reshape(-1,1)
        
        dx4x = torch.autograd.grad(dx4, x , grad_outputs=torch.ones_like(dx4) ,\
                                  create_graph = True)
        dx4x1 = dx4x[0][:,1].reshape(-1,1)
        dx4x2 = dx4x[0][:,2].reshape(-1,1)
        dx4x3 = dx4x[0][:,3].reshape(-1,1)
        dx4x4 = dx4x[0][:,4].reshape(-1,1)
        dx4x5 = dx4x[0][:,5].reshape(-1,1)
        dx4x6 = dx4x[0][:,6].reshape(-1,1)
        dx4x7 = dx4x[0][:,7].reshape(-1,1)
        
        dx5x = torch.autograd.grad(dx5, x , grad_outputs=torch.ones_like(dx5) ,\
                                  create_graph = True)
        dx5x1 = dx5x[0][:,1].reshape(-1,1)
        dx5x2 = dx5x[0][:,2].reshape(-1,1)
        dx5x3 = dx5x[0][:,3].reshape(-1,1)
        dx5x4 = dx5x[0][:,4].reshape(-1,1)
        dx5x5 = dx5x[0][:,5].reshape(-1,1)
        dx5x6 = dx5x[0][:,6].reshape(-1,1)
        dx5x7 = dx5x[0][:,7].reshape(-1,1)
        
        
        dx6x = torch.autograd.grad(dx6, x , grad_outputs=torch.ones_like(dx6) ,\
                                  create_graph = True)
        dx6x1 = dx6x[0][:,1].reshape(-1,1)
        dx6x2 = dx6x[0][:,2].reshape(-1,1)
        dx6x3 = dx6x[0][:,3].reshape(-1,1)
        dx6x4 = dx6x[0][:,4].reshape(-1,1)
        dx6x5 = dx6x[0][:,5].reshape(-1,1)
        dx6x6 = dx6x[0][:,6].reshape(-1,1)
        dx6x7 = dx6x[0][:,7].reshape(-1,1)
        
        dx7x = torch.autograd.grad(dx7, x , grad_outputs=torch.ones_like(dx7) ,\
                                  create_graph = True)
        dx7x1 = dx7x[0][:,1].reshape(-1,1)
        dx7x2 = dx7x[0][:,2].reshape(-1,1)
        dx7x3 = dx7x[0][:,3].reshape(-1,1)
        dx7x4 = dx7x[0][:,4].reshape(-1,1)
        dx7x5 = dx7x[0][:,5].reshape(-1,1)
        dx7x6 = dx7x[0][:,6].reshape(-1,1)
        dx7x7 = dx7x[0][:,7].reshape(-1,1)
        
        
        
        if len(x) == 0:
            print('zero batch size for domain!')
            #DO = torch.tensor(0).cuda().float()
        DO = ( dt + self.mu(x[:,1])*( dx1 ) + self.mu(x[:,2])*( dx2 ) + self.mu(x[:,3])*( dx3 ) + self.mu(x[:,4])*( dx4 ) + self.mu(x[:,5])*( dx5 ) + self.mu(x[:,6])*( dx6 ) + self.mu(x[:,7])*( dx7 ) \
                  + 0.5*(    1*(self.sigma(x[:,1])*self.sigma(x[:,1]))*dx1x1  \
                                + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,2]))*dx1x2  \
                                    + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,3]))*dx1x3  \
                                        + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,4]))*dx1x4  \
                                            + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,5]))*dx1x5  \
                                                + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,6]))*dx1x6  \
                                                    + self.RU*(self.sigma(x[:,1])*self.sigma(x[:,7]))*dx1x7  \
                            + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,1]))*dx2x1  \
                                + 1*(self.sigma(x[:,2])*self.sigma(x[:,2]))*dx2x2  \
                                    + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,3]))*dx2x3  \
                                        + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,4]))*dx2x4  \
                                            + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,5]))*dx2x5  \
                                                + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,6]))*dx2x6  \
                                                    + self.RU*(self.sigma(x[:,2])*self.sigma(x[:,7]))*dx2x7  \
                            + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,1]))*dx3x1  \
                                + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,2]))*dx3x2  \
                                    + 1*(self.sigma(x[:,3])*self.sigma(x[:,3]))*dx3x3  \
                                         + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,4]))*dx3x4  \
                                             + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,5]))*dx3x5  \
                                                 + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,6]))*dx3x6  \
                                                     + self.RU*(self.sigma(x[:,3])*self.sigma(x[:,7]))*dx3x7  \
                            + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,1]))*dx4x1  \
                                + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,2]))*dx4x2  \
                                    + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,3]))*dx4x3  \
                                         + 1*(self.sigma(x[:,4])*self.sigma(x[:,4]))*dx4x4  \
                                             + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,5]))*dx4x5  \
                                                 + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,6]))*dx4x6  \
                                                     + self.RU*(self.sigma(x[:,4])*self.sigma(x[:,7]))*dx4x7  \
                            + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,1]))*dx5x1  \
                                + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,2]))*dx5x2  \
                                    + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,3]))*dx5x3  \
                                         + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,4]))*dx5x4  \
                                             + 1*(self.sigma(x[:,5])*self.sigma(x[:,5]))*dx5x5  \
                                                 + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,6]))*dx5x6  \
                                                     + self.RU*(self.sigma(x[:,5])*self.sigma(x[:,7]))*dx5x7  \
                            + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,1]))*dx6x1  \
                                + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,2]))*dx6x2  \
                                    + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,3]))*dx6x3  \
                                         + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,4]))*dx6x4  \
                                             + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,5]))*dx6x5  \
                                                 + 1*(self.sigma(x[:,6])*self.sigma(x[:,6]))*dx6x6  \
                                                     + self.RU*(self.sigma(x[:,6])*self.sigma(x[:,7]))*dx6x7  \
                            + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,1]))*dx7x1  \
                                + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,2]))*dx7x2  \
                                    + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,3]))*dx7x3  \
                                         + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,4]))*dx7x4  \
                                             + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,5]))*dx7x5  \
                                                 + self.RU*(self.sigma(x[:,7])*self.sigma(x[:,6]))*dx7x6  \
                                                     + 1*(self.sigma(x[:,7])*self.sigma(x[:,7]))*dx7x7  \
                                                       ) - self.R*self.net(x) )**2
     
        # Terminal Condition
        TC = ( self.g(x_terminal) - self.net(x_terminal))**2 
        
        # Boundry Condition
        # len() is safe here , because it just shows batch number 
        if( len(x_boundry) != 0):
            BC = torch.max( self.g(x_boundry) - self.net(x_boundry) , torch.zeros([len(x_boundry),1]).cuda() )**2 
        else:
            #print('zero batch size for outside domain!')
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