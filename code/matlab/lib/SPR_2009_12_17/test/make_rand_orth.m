function	X = make_rand_orth(M,N)
% Make M x N random normalized matrix
% X = make_nn_diag(N,K)

if nargin==1, N=M; end;

X = randn(M,N);
X = repmultiply(X, 1./sqrt(sum(X.^2,2)));
%X = X + 0.1*eye(M,N);
