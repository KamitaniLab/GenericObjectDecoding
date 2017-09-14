function	[A,W,SW,S,Q,ix_act] = ...
				compo_update_act(XX,A,W,SW,S,Q,SY,i_up,jx,ix_act)
% Update component
% XX = X(ix_act,:)*X'  [M_act x M_all]

% New alpha
a0 = A(i_up);
s0 = S(i_up);
b0 = 1 - s0./a0;	% scale factor for Q & S
qq0  = SY*sum(Q(:,i_up).^2,1);
anew = s0.^2 ./(qq0 - s0.*b0);

%if (anew - a0) < eps, return; end;

A(i_up) = anew;

% Weight  & variance update
ss = 1./(SW(jx,jx) + 1./(anew - a0));
SX = SW(jx,:);
SW = SW - ss * (SX' * SX);
SW = 0.5 * (SW + SW');	% guarantee symmetric matrix
Wj = W(:,jx) * ss;
W  = W - Wj * SX;

% Residual correlation
% XX = X(ix_act,:)*X'  [M_act x M_all]
P  = SX * XX;
S  = abs(S + ss * P.^2);
Q  = Q + Wj * P;
return
