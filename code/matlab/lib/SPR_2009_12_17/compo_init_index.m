function	[SY,ix] = compo_init_index(xx,YX,YY,T)
%
% --- Initialization: Find input component with maximum correlation
% 
% N    # of output
% M    # of input
% T    Total number of data
% xx = diag(XX)'	% 1 x M_all

[N, M_all] = size(YX);

R  = sum(YX.^2,1) ./xx; % (YX)^2/(XX) : correlation of Y & X
[Rmax, ix] = max(R);
ix = ix(1);

% Residual error in MSE estimation
dYY = abs(sum(YY) - Rmax);	% = (Y - W*X(ix))^2 = Y^2 - (YX)^2/(XX)
SY  = (N*T)./abs(dYY);
