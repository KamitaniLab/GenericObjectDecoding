function	[A,W,SW,S,Q] = compo_init_alpha(XX,YX,xx,ix)
% --- Initialization
% XX = X(ix,:)*X'  % 1 x M_all
% YX = Y * X'		% N x M_all
% xx = diag(XX)'	% 1 x M_all

M_all = length(xx);

% alpha_ini = 1
A  = zeros(1,M_all);
a0 = 1;
A(ix) = a0;

SW = 1/(a0 + xx(ix));		% 1 x 1
W  = YX(:,ix)*SW;			% N x 1
S  = abs(xx - SW * XX.^2) ;	% 1 x M_all
Q  = YX -  W * XX;			% 1 x M_all
