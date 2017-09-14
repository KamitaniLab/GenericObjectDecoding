function	[Ydev] = predict_variance(X, Model, parm)
% Variance of posterior predictive distribution
%  [Ydev] = predict_variance(X, Model, parm)
%  
%  
%  
% 
% 2009-11-5 M. Sato


if ~exist('parm','var'), parm = []; end;

if ~isfield(parm,'Tau')
	tau = 1 ; 
	D   = 1 ; 
else
	tau = parm.Tau;
	D   = parm.Dtau;
end


[M, Tx, Ntrial] = size(X);
[N, MD] = size(Model.W );
T = Tx - (D-1)*tau;

if	isfield(Model,'ix_act')
	Flag = zeros(M*D,1);
	% active index for embeded space
	Flag(Model.ix_act) = 1;
	% extract active input dimension
	ix = find( sum(reshape(Flag,[M,D]),2) > 0);
	X  = X(ix,:,:);
end

[M, Tx, Ntrial] = size(X);

SY = Model.SY;

if length(SY)==1, SY = repmat(SY,[N 1]); end;

SW   = Model.SW; % = inv(Tall*XX + diag(Ainv))

Tdelay = tau*(D-1);
indxT = (1:T) + Tdelay;
indxD = 0:tau:Tdelay;
indx  = repmat( -indxD(:),[1 T]) + repmat(indxT,[D 1]);

methods = {...
	'linear_sparse_space';
	'linear_sparse_stepwise';
	'linear_sparse_cov';
	};

method_id = strmatch(Model.method,methods);

switch	method_id
case	1
	%  'linear_sparse_space';
	% E = (Y-W*X)^2/SY +  W^2 * (1/A) /SX
	% SY  = SX/rx = (D + 1)*SX,  rx  = 1/(D + 1);
	%  <(W-<W>)*(W-<W>)> = SX*SW = SY*SW/(D+1)
	%	SW  = inv(Tall*XX + diag(Ainv))
	% 2D-matrix by reshape
	X   = reshape(X,[M, Tx*Ntrial]);
	XSX = sum(X .* (SW*X) ,1);
	XSX = reshape(XSX,[Tx ,Ntrial]);
	XSX = sum(reshape(XSX(indx,:),[D T*Ntrial]),1)/(D+1) + 1;
	
	Ydev = SY * XSX;
case	2
	%	'linear_sparse_stepwise'
	% E = ( (Y-W*X)^2 +  W^2 * (1/A) )/SY
	% SW = 1./(Tall*XX + Ainv) : 1 x Xdim*D
	% Input variance
	XX  = reshape(X(:,indx(:),:).^2, [M*D, T*Ntrial]);% Xdim*D x T
	XSX = sum(repmultiply(XX,SW(:)), 1) + 1;
	
	Ydev = SY * XSX;
case	3
	%  'linear_sparse_cov'
	%	SW  = inv( XX + diag(1./(A*T)) );  % M x M
	
	X  = reshape(X(:,indx(:)), [M*D, T]);% Xdim*D x T
	XSX = sum(X .* (SW*X) ,1) + 1;
	
	Ydev = SY * XSX;
case	4
	%	SW = (1/T)*(SX./XXX);	% ( N x M )
	Ydev = zeros(N,T);
	XX = X.^2;
	for n=1:N
		SWn = repmat(SW(n,:)', [1 T]);
		Ydev(n,:) = sum((SWn .* XX), 1);
	end
	Ydev = sqrt( repmat(SY,[1 T]) + Ydev );
end

%ysize=size(Ydev)
Ydev = sqrt(Ydev);

% Scale back output according to output variance (parm.ynorm)
if isfield(parm,'ynorm') && ~isempty(parm.ynorm)
	Ydev = repmultiply(Ydev , parm.ynorm);
end

Ydev = reshape(	Ydev ,[N, T, Ntrial]);
