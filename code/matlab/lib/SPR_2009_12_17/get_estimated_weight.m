function	[W ,Ydim, Xdim, Dtau] = get_estimated_weight(Model,parm,wmode);
% return Weight matrix from 'Model'
%  [W ,Ydim, Xdim, Dtau] = get_estimated_weight(Model,parm,wmode);
% W : Weight matrix : 2D-matrix [Ydim x (Xdim * Dtau)]
% Dtau : time embedding dim
% Ydim : Output space dim
% Xdim : Input space dim
%   To get temporal filter shape of Weight matrix
% Wd = reshape( W, [Ydim, Xdim, Dtau]);
% Wd(n,m,:) : Weight for n-th output & m-th input data
%
% 2008-5-20 Masa-aki Sato

if isfield(parm,'Dtau')
	Dtau  = parm.Dtau;
else
	Dtau  = 1;
end

M_all = Model.M_all;
Ydim  = size(Model.W,1);
Xdim  = M_all/Dtau;

if isfield(Model,'ix_act')
	% Active index
	ix_act = Model.ix_act;

	W = zeros(Ydim ,M_all);
	W(:,ix_act) = Model.W;
else
	W =  Model.W;
end

if exist('wmode','var') && wmode==0, return; end;

if length(parm.xmean) == Xdim,
	parm.xmean = repmat(parm.xmean ,[Dtau 1]);
	parm.xnorm = repmat(parm.xnorm ,[Dtau 1]);
end

% Scale back by normalization factor
if isfield(parm,'xnorm') & isfield(parm,'ynorm')
	W = (parm.ynorm(:)*(1./parm.xnorm(:)')) .* W;
end
