function	X = make_nn_diag(N,K)
% Make N x N matrix with K-principal dimension
% X = make_nn_diag(N,K)
Adiag = 0.2;

Z = rand(N,K);
Z = repmultiply( Z, 1./sqrt(sum(Z.^2,1)) );
X = Adiag * eye(N) + Z * Z';

return
%%%--- END ---%%%

% Make N x N matrix whose n-th diag elements have correlation with neighbor
% X(n,n+k) = 1, k=-K:K

X  = speye(N,N);

%ix = randperm(N);
%ix = ix(1:K);

Nd = max(fix(N/K),1);
ix = 1:Nd:N;

for n = ix
	%a = exp(-(n/NN).^2);
%	X1= spdiags(a*ones(N,1),n,N,N);
	X1= spdiags(rand(N,1),n,N,N);
%	X = X + X1 ;
	X = X + X1 + X1';
end

X = X ./ repmat( 1./sqrt(sum(X.^2,2)) , [1, N]);
