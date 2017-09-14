function	W = get_estimated_weight2(Model,parm,wmode);
% Return Full Weight matrix from 'Model'
%  W = get_estimated_weight2(Model,parm);
%  W = get_estimated_weight2(Model,parm,wmode);
% --- input
% Model.W : Weight matrix for active input [Ydim x Nactive]
% Model.ix_act : active input index
% parm.Dtau
% parm.M_all
% parm.xnorm
% parm.ynorm
%  In the default mode, normalization constants, xnorm and ynorm
%  are scaled back into the weight matrix 'W' : weight for original input
%  If wmode is given and wmode = 0, 
%  no scale normalization is done: weight for normalized input
%  
% --- output
% W : Weight matrix : 3D-array [Ydim x Xdim x Dtau)]
% W(n,m,:) : temporal weight for n-th output & m-th input data
% Dtau : time embedding dim
% Ydim : Output space dim
% Xdim : Input space dim
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

if ~exist('wmode','var') || wmode~=0,
	if length(parm.xmean) == Xdim,
		parm.xmean = repmat(parm.xmean ,[Dtau 1]);
		parm.xnorm = repmat(parm.xnorm ,[Dtau 1]);
	end
	
	% Scale back by normalization factor
	if isfield(parm,'xnorm') & isfield(parm,'ynorm')
		W = (parm.ynorm(:)*(1./parm.xnorm(:)')) .* W;
	end
end

W = reshape( W, [Ydim, Xdim, Dtau]);
