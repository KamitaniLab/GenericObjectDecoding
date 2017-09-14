%	X : M x Tx x Nr
%	Y : N x T  x Nr
%  dY : N x T  x Nr
% dYX : N x (M*D)
%  W  : N x (M*D)

% Xdelay(t) = [X(t+tau*(D-1)); ... ; X(t+tau); X(t)]

% Error calculation in time embeded space
%   dY = Y - Xdelay * W
[dY] = error_delay_time(X,Y,W,T,N,M,D,tau);

% Correlation of dY and X in time embeded space
%   dYX = dY * Xdelay';
dYX = error_corr_delay(X,dY,D,tau)

% Component-wise weight update of Error function
%	  E = (Y - W*X)^2/SY + A*W^2
W = weight_update_iter(X,dY,W,XX,A,SY) 
%  dW(n,m) = (dYX(n,m) - W(n,m)*A(n,m)*S(m))./ (XX(m) + A(n,m)*S(m));
%	X : M x T
%  XX : M x 1
%  SY : N x 1
%  dY : N x T
%  W  : N x M
%  A  : N x M

% Component-wise weight update of Error function in time embeded space
%	  E = (Y - W*X)^2 + W*A*W
[W] = weight_update_embed(X,dY,W,XX,A,tau);
% dW(n,m) = (dYX(n,m) - W(n,m)*A(m))./ (XX(m) + A(m));
%	X : M x Tx x Nr
%  dY : N x T  x Nr
%  W  : N x (M*D)
%  XX : M x 1 or 1 x M
%  A  : M x 1 or 1 x M

% When A is the same size as W
W  = weight_update_embed_aa(X,dY,W,XX,A,tau);
% dW(n,m) = (dYX(n,m) - W(n,m)*A(n,m))./ (XX(m) + A(n,m));
%  A  : N x (M*D)

%  Output calculation in time embeded space
%  Y = W * Xdelay
Y = weight_out_delay_time(X,W,T,tau);  

% When # of sample is varied in each trial
%  Ntr(n) : # of samples in n-th trial
dY  = error_delay_time_sw(X,Y,W,tau,Ntr);
dYX = error_corr_delay_sw(X,dY,D,tau,Ntr)
W   = weight_update_embed_sw(X,dY,W,XX,A,tau,Ntr)
