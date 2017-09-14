function  [A,W,SW,S,Q,ix_act] = ...
			compo_delete_act(XX,A,W,SW,S,Q,SY,i_del,jx,ix_act)
% Delete component
% XX = X(ix_act,:)*X'  [M_act x M_all]

A(i_del) = 0;

M  = length(ix_act);
jj = [1:(jx-1), (jx+1):M];

% New active index
ix_act = ix_act(jj);

% Weight  & variance update
ss = 1./SW(jx,jx);
SX = SW(jx,jj);
SW = SW(jj,jj) - ss * (SX' * SX);
SW = 0.5 * (SW + SW');	% guarantee symmetric matrix
Wj = W(:,jx) * ss;
W  = W(:,jj) - Wj * SX;

% Residual correlation
P  = SX * XX(jj,:);
S  = abs(S + ss * P.^2);
Q  = Q + Wj * P;
return
