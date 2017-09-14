function	[Y] = predict_output(X, Model, parm)
% Prediction from input data
% Input X should be normalized according to 'parm.data_norm' mode
% Time delay embedding for X is done in this function
%  Y = predict_output(X, Model)
%    = Model.W * X
%  Y = predict_output(X, Model, parm)
%    = (Model.W * X) * (parm.ynorm) + (parm.ymean)
% --- Input
%  X  : Input data  ( M x T x Ntrial)
%  M  =  # of input (original input space dimension)
%  T  =  # of time sample
% parm.data_norm : normalization mode
% parm.Tau   = Lag time
% parm.Dtau  = Number of embedding dimension
% parm.ymean = training data output mean
% parm.ynorm = training data output variance (standard deviation)
%
% 
% --- Output
%  Y  : Output data ( N x (T- (D-1)*tau) x Ntrial )
%  N  =  # of output
% 
% 2006/1/27 M. Sato
% 2008-5-8 Modified by M. Sato

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
	
	% adjust size of W
	W = zeros(N,M*D);
	W(:,Model.ix_act) = Model.W;
	W = reshape(W,[N,M,D]);
	W = W(:,ix,:);
	
	M = length(ix);
	W = reshape(W,[N,M*D]);
else
	W = Model.W;
end

% Embedding dimension
% D = size(W,2)/size(X,1)

% Output from normalized input
if isempty(W) % modified by TH130111
    Y=rand(1,size(X,2))*1e-100;
else
    Y = weight_out_delay_time(X, W, T, tau);
end
% Scale back output according to output variance (parm.ynorm)
if isfield(parm,'ynorm') && ~isempty(parm.ynorm)
	Y = Y .* repmat(parm.ynorm, [1 T Ntrial]);
end

if parm.data_norm == 2, return; end;

% Add original output mean (parm.ymean)
if isfield(parm,'ymean') && ~isempty(parm.ymean)
	Y = Y + repmat(parm.ymean, [1 T Ntrial]);
end
