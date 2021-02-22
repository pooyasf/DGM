% https://www.mathworks.com/help/fininst/basketsensbyls.html

Settle = 'May-1-2009';
Maturity  = 'May-1-2011';

% Define RateSpec
Rate = -0.02;
Compounding = -1;
RateSpec = intenvset('ValuationDate', Settle, 'StartDates',...
Settle, 'EndDates', Maturity, 'Rates', Rate, 'Compounding', Compounding);

% Define the Correlation matrix. Correlation matrices are symmetric, 
% and have ones along the main diagonal.
NumInst  = 7;
InstIdx = ones(NumInst,1);
Corr = diag(ones(NumInst,1), 0);
idx = find(~eye(size(Corr)));
Corr(idx) = Corr(idx) + 0.75;

% Define BasketStockSpec
AssetPrice =  [1; 1; 1; 1; 1; 1; 1;]; 
Volatility = 0.25;
Quantity = [1/7; 1/7; 1/7; 1/7; 1/7; 1/7; 1/7;];
BasketStockSpec = basketstockspec(Volatility, AssetPrice, Quantity, Corr);

% Compute the price of the put basket option. Calculate also the delta 
% of the first stock.
OptSpec = {'call'};
Strike = 1;
OutSpec = {'Price'}; 

                                     
PriceSens = basketsensbyls(RateSpec, BasketStockSpec, OptSpec,...
Strike, Settle, Maturity,'OutSpec', OutSpec, 'AmericanOpt' , 1  , 'NumTrials' , 5000)
