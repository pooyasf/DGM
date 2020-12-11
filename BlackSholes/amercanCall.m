% Matlab Program 10: Compares European and American Call options by using an
% explicit method Parameters of the problem:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r=0.;
 % Interest rate
sigma=0.25;
 % Volatility of the underlying
d=0.02;
 % Continuous dividend yield
M=1600;
 % Number of time points
N=100;
 % Number of share price points
Smax=2;
 % Maximum share price considered
Smin=0;
 % Minimum share price considered
T=2.;
 % Maturation (expiry)of contract
E=1;
 % Exercise price of the underlying
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt=(T/M);
 % Time step
ds=(Smax-Smin)/N; % Price step
% Initializing the matrix of the option values: v is the European and vam is the American option
v(1:N,1:M) = 0.0;
vam(1:N,1:M) = 0.0;
% Initial conditions prescribed by the Call payoff at expiry: V(S,T)=max(E-S,0)
v(1:N,1)=max((Smin+(0:N-1)*ds-E),zeros(size(1:N)))';
vam(1:N,1)=max((Smin+(0:N-1)*ds-E),zeros(size(1:N)))';
% Boundary conditions prescribed by Call Options with dividends:
% V(0,t)=0
v(1,2:M)=zeros(M-1,1)';
vam(1,2:M)=zeros(M-1,1)';
% V(S,t)=Se^(-d*(T-t))-Ee^(-r(T-t)) as S -> infininty.
v(N,2:M)=((N-1)*ds+Smin)*exp(-d*(1:M-1)*dt)-E*exp(-r*(1:M-1)*dt);
vam(N,2:M)=((N-1)*ds+Smin)*exp(-d*(1:M-1)*dt)-E*exp(-r*(1:M-1)*dt);
% Determining the matrix coeficients of the explicit algorithm
aa=0.5*dt*(sigma*sigma*(1:N-2).*(1:N-2)-(r-d)*(1:N-2))';
bb=1-dt*(sigma*sigma*(1:N-2).*(1:N-2)+r)';
cc=0.5*dt*(sigma*sigma*(1:N-2).*(1:N-2)+(r-d)*(1:N-2))';
% Implementing the explicit algorithm
for i=2:M,
v(2:N-1,i)=bb.*v(2:N-1,i-1)+cc.*v(3:N,i-1)+aa.*v(1:N-2,i-1);
% Checks if early exercise is better for the American Option
vam(2:N-1,i)=max(bb.*vam(2:N-1,i-1)+cc.*vam(3:N,i-1)+aa.*vam(1:N-2,i-1),vam(2:N-1,1));
end
% Reversal of the time components in the matrix as the solution of the Black-Scholes
% equation was performed backwards
v=fliplr(v);
vam = fliplr(vam);
% Compares the value today of the European (blue) and American (red) Calls,V(S,t), as a function of S.
% The green curve represents the payoff at expiry.
plot(Smin+ds*(0:(N-2)),v(1:(N-1),M)','g-',Smin+ds*(0:(N-2)),v(1:(N-1),1)','b-',Smin+ds*(0:(N-2)),vam(1:(N-1),1)','r-');
xlabel('S');
ylabel('V(S,t)');
title('European (blue) and American (red) Call Options');
