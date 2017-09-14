function	[W ,ix_act] = get_embed_weight(Model,parm);
% return Weight matrix from 'Model'
%  [W ,ix_act] = get_embed_weight(Model,parm);
% --- input
% Model.W : Weight matrix for active input [Ydim x Nactive]
% Model.ix_act : active input index
% parm.Dtau
% parm.M_all
% --- output
% W : Weight matrix for normalized input : [Ydim x Xeff x Dtau]
%     only active input component is returned
% Dtau : time embedding dim
% Ydim : Output space dim
% Xeff : Effective input space dim
% ix_act : index for Effective input [1 x Xeff]
%
% 2008-5-20 Masa-aki Sato

if isfield(parm,'Dtau')
	Dtau  = parm.Dtau;
else
	Dtau  = 1;
end

W =  Model.W;

if isfield(Model,'ix_act')
	% Active index
	ix_act = Model.ix_act;
	M_all  = length(ix_act);
else
	M_all  = Model.M_all;
	ix_act = 1:M_all;
end

Ydim  = size(Model.W,1);
Xeff  = M_all/Dtau;

W = reshape(W, [Ydim,Xeff,Dtau]);
ix_act = ix_act(1:Xeff);
