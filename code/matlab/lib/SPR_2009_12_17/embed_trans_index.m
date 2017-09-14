function [ix_dim,ix_tau,jx_act] = embed_trans_index(ix_act,M,D,mode)

if ~exist('mode','var'), mode = 0; end;

flg = zeros(M,D);
flg(ix_act) = 1;

% find active X-dim
ix_dim = find( sum(flg,2) > 0 );

% find active t-dim
if mode==0
	ix_tau = find( sum(flg,1) > 0 );
else
	ix_tau = 1:D;
end

% find active index in extracted reduced space
flg = flg(ix_dim,ix_tau);

jx_act = find( flg(:) > 0);
