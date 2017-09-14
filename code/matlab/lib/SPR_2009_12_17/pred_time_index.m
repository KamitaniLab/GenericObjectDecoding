function	[tx,ty,T] = pred_time_index(x,parm)
%  Time index for delay embedding prediction
%  [tx,ty,T] = pred_time_index(x,parm)
% --- Input
% x : Input data  : Xdim x Tsample x Ntrial
% parm : Time delay embedding parameter
% parm.Tau       = Lag time
% parm.Dtau      = Number of embedding dimension
% parm.Tpred     = Prediction time step :
%                  y(t+Tpred) = W * [x(t); ...; x(t-Tdelay)]
%                  ydata(t)   = W * xdata(t)
% --- Output
%	%  Time period for prediction
%	T  = Ndata - Tpred - Tdelay; 
%	tx = 1:(Ndata-Tpred);
%	ty = (Tpred+Tdelay+1):Ndata;
%
% X(1:Tdelay+1) -> Y(Tdelay + Tpred + 1)
%
% ypred(:,1:T)  =   y(:,ty)
% xpred(:,1:T+Tdelay) = x(:,tx);
%
% 2007/1/15 Made by M. Sato

[Xdim,Ndata,Ntrial] = size(x);

if ~exist('parm','var'), 
	T = Ndata;
	tx = 1:T;
	ty = tx;
	return
end;

if ~isfield(parm,'Tpred')
	parm.Tpred = 0 ; 
end

if ~isfield(parm,'Tau')
	% Setting for function mapping ( No embedding case )
	parm.Tau   = 1 ; 
	parm.Dtau  = 1 ; 
end

if parm.Dtau == 1, parm.Tau   = 1 ; end;

% Time delay embedding parameter
Tpred = parm.Tpred ;
Tau   = parm.Tau   ;
D     = parm.Dtau  ;
Tdelay = Tau*(D-1);

if D == 1 && Tpred == 0,
	% No delay embedding case
	T = Ndata;
	tx = 1:T;
	ty = tx;
	return
end;
%
% ---  Prepare embedding input
%
if Tpred >= 0,
	% Forward Prediction 
	% X(1:Tdelay+1) -> Y(Tdelay + 1 + Tpred) = Y(TPRED + 1)
	TPRED = Tpred + Tdelay;
	
	%  Time period for prediction
	T  = Ndata - TPRED; 
	tx = 1:(Ndata-Tpred);
	ty = (TPRED+1):Ndata;
elseif (Tpred < 0) && (Tpred >= -Tdelay),
	% Prediction time is included in input period
	Tpred = -Tpred; % > 0

	% X(1:Tdelay+1) -> Y(Tdelay + 1 - Tpred) = Y(TPRED + 1)
	TPRED = Tdelay - Tpred;
	
	%  Time period for prediction
	T  = Ndata - Tdelay; 
	tx = 1:Ndata;
	ty = (TPRED+1):(Ndata-Tpred);
else
	% Backward Prediction 
	Tpred = - Tpred; % > Tdelay

	% X((TPRED+1):Tpred+1) -> Y(1)
	TPRED = Tpred - Tdelay;
	
	%  Time period for prediction
	T  = Ndata - Tpred; 
	tx = (TPRED+1):Ndata;
	ty = 1:T;
end
