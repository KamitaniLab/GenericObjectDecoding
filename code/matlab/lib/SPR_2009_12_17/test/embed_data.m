function	[xdata,ydata,tx,ty] = embed_data(x,y,parm)
% Delay or Integral-delay embedding for input-output prediction
%  [xdata,ydata,tx,ty] = embed_data(x, y,  parm)
%  [xdata,dumy, tx ]   = embed_data(x, [], parm)
% --- Input
% x : Input data  : Xdim x Tsample x Ntrial
% y : Output data : Ydim x Tsample x Ntrial
% parm : Time delay embedding parameter
% parm.Tau       = Lag time
% parm.Dtau      = Number of embedding dimension
% parm.Tpred     = Prediction time step :
%                  y(t+Tpred) = W * [x(t); ...; x(t-Tdelay)]
%                  ydata(t)   = W * xdata(t)
% --- Output
% ydata(:,1:T) =   y(:,ty)
% xdata(:,1:T) = [ x(:,tx); x(:,tx-tau); ...; x(:,tx-tau*(Dtau-1) ]
%
%    tx = (1:T) + Tdelay;
%    ty = tx + Tpred
%    Tdelay  = tau*(Dtau-1);
%    Tsample = T + Tpred + Tdelay
%
% 2007/1/15 Made by M. Sato

[Xdim,Ndata,Ntrial] = size(x);

if ~exist('parm','var'), parm = []; end;

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
elseif (Tpred < 0) & (Tpred >= -Tdelay),
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
	TPRED = - Tpred; % > Tdelay

	% X((Tpred+1):TPRED+1) -> Y(1)
	Tpred = TPRED - Tdelay;
	
	%  Time period for prediction
	T  = Ndata - TPRED; 
	tx = (Tpred+1):Ndata;
	ty = 1:T;
end

[xdata ,t] = delay_embed(x(:,tx,:),Tau,D,T);
tx = tx(t);

if ~exist('y','var') | isempty(y),
	ydata = [];
else
	ydata = y(:,ty,:);
end

%parm.T     = T ; 
%parm.Ndata = Ndata;
