function	[A,W,SW,S,Q,ix_act] = ...
				compo_add_act(XX,Xadd,A,W,SW,S,Q,SY,i_add,ix_act)
% Add component
% XX   = X(ix_act,:)*X'  [M_act x M_all]
% Xadd = X(i_add,:) *X'  [1 x M_all]

% New alpha
anew = S(i_add).^2 ./(SY*sum(Q(:,i_add).^2,1) - S(i_add));
A(i_add) = anew;

% Weight  & variance update
ss = 1./(S(i_add) + anew);
SX = - Xadd(ix_act) * SW;
Sj = (ss * SX);
SW = [SW + ss * (SX' * SX), Sj' ; Sj , ss];
SW = 0.5 * (SW + SW');	% guarantee symmetric matrix
Wj = ss * Q(:,i_add);
W  = [W + Wj * SX , Wj];

% Residual correlation
P  = SX * XX + Xadd;
S  = abs(S - ss * P.^2);
Q  = Q - Wj * P;

% New active index
ix_act = [ix_act(:) ; i_add];
return
