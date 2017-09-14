function	d = log_det(sxx)
% Log( det(sxx) ) for Positive definite matrix
% 共分散行列は常に正定値対称行列

% Gauss の消去法による三角行列因子
[L,U]	= lu(sxx);
eig_val = abs( diag(U) );
eig_val = eig_val(eig_val > 0);
d		= sum( log( eig_val ) );

% コレスキ分解 : X が正定行列のとき，三角分解より早い
%R = chol(X) は、X が正定行列のとき、R'*R = X となる上三角行列 R 

%U = chol(sxx);
%d = 2*sum(log(diag(U)));

