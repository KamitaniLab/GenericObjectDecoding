function	[Wout,Weff] = make_test_weight(parm)
% Make output weigtht for test
% -Input
% parm.Ydim   =  Output dim
% parm.Xdim   =  Input dim
% parm.Meff   =  Effective Input dim
% parm.Tau    =  Lag time steps
% parm.Dtau   =  Number of embedding dimension
% parm.WXmax = maximum weight of X-dim
% parm.WYmax = bias weight for Y-dim
% parm.Wtau  = effctive time delay length of weight
% -Output
% Wout  : Weight for time embedded input [Ydim, Xdim*Dtau]
% Weff  : nonzero weight for [Ydim, Meff*Dtau]
%         effective input dimension : [Meff x Wtau]

% Input dim
if isfield(parm,'Xdim')
	Xdim = parm.Xdim ;
else
	Xdim = 100;
end
% Output dim
if isfield(parm,'Ydim')
	Ydim = parm.Ydim ;
else
	Ydim = 1;
end
% Effective Input dim
if isfield(parm,'Meff')
	Meff = parm.Meff ;
else
	Meff = 10; ;
end
% Number of embedding dimension
if isfield(parm,'Dtau')
	Dtau  = parm.Dtau  ;
else
	Dtau  = 1;
end

if Meff > Xdim, Meff = Xdim; end;
if Dtau==0, Dtau=1; end;

% effctive time delay length of weight
if isfield(parm,'Wtau')
	tau  = parm.Wtau  ;
else
	tau  = Dtau;
end
% maximum weight of X-dim
if isfield(parm,'WXmax')
	WXmax = parm.WXmax;
else
	WXmax = 100;	
end
% maximum bias weight for Y-dim
if isfield(parm,'WYmax')
	WYmax = parm.WYmax;
else
	WYmax = 10;	
end

%
% --- Make Output Weight matrix
%
% Output weight

% Spatial weight
Wbase = rand(Ydim,Meff)*WXmax + WYmax;

% Temporal filter
t = (0:tau-1)/max((tau-1)/2,1);
Wtau  = exp( - 2* t.^2);

ix_eff = 1:Meff;
Weff   = zeros(Ydim,Meff*Dtau);
Wout   = zeros(Ydim,Xdim*Dtau);

for n = 1:tau
	Weff(:,ix_eff + Meff*(n-1)) = Wtau(n) * Wbase;
	Wout(:,ix_eff + Xdim*(n-1)) = Wtau(n) * Wbase;
end
