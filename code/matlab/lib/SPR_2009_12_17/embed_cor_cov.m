function	Xcov = embed_cor_cov(X,D,tau,ix_act)
% Partial covariance matrix in time delay embedding space
%  XX = embed_cor_cov(X,D,tau,ix_act)
%  XX = X(ix_act,:)*X'
% --- Input
%  X : input signal : [Xdim x Tsample x Ntrial]
%  D : time embedding dimension  (integer)
%  tau : delay time in sample number (integer)
% ix_act : index in delay embedding space
% --- Output
%  XX(:,[j,t2]) = Xtau(ix_act,:) * Xtau([j,t2],:)'
%  Xtau([i,t0],t) = X(i ,t + tau*(D - t0)) : [(Xdim*D) x T x Ntrial]
%  T = Tsample - tau*(D-1)
%
% 2008-5-22 Masa-aki Sato

[M ,Tall, Nr] = size(X);

[ix_dim,ix_tau,jx_act] = embed_trans_index(ix_act,M,D,0);

Tdelay = tau*(D-1);
T = Tall - Tdelay;

% Time delayed embedding index
Tbgn = Tdelay + 1 - (0:tau:Tdelay);
Tend = Tbgn + T - 1;

% X-dim index
Iend = (1:D)*M;
Ibgn = Iend - M + 1;

NX = length(ix_dim);
NT = length(ix_tau);

Xcov = zeros(NX*NT,M*D);

ix = 1:NX;

for n=1:NT
	% delay index 'i'
	i = ix_tau(n);
	
	for j=1:D
		% delay index 'j'
		Xcov(ix, Ibgn(j):Iend(j)) = ...
			reshape(X(ix_dim,Tbgn(i):Tend(i),:), NX,T*Nr) ...
		  * reshape(X(:,Tbgn(j):Tend(j),:), M,T*Nr)';
		   
	end
	
	ix = ix + NX;
end

Xcov = Xcov(jx_act,:);
