function	Xcov = embed_covariance(X,D,tau)
% Covariance matrix in time delay embedding space
%  Xcov = embed_covariance(X,D,tau)
% --- Input
%  X : input signal : [Xdim x Tsample x Ntrial]
%  D : time embedding dimension  (integer)
%  tau : delay time in sample number (integer)
% --- Output
%  Xcov([i,t1],[j,t2]) = Xtau([i,t1],:) * Xtau([j,t2],:)'
%  Xtau([i,t0],t) = X(i ,t + tau*(D - t0)) : [(Xdim*D) x T x Ntrial]
%  T = Tsample - tau*(D-1)
%
% 2008-5-22 Masa-aki Sato

[N ,Tall, Nr] = size(X);

Tdelay = tau*(D-1);

T = Tall - Tdelay;

% Time delayed embedding index
Tbgn = Tdelay + 1 - (0:tau:Tdelay);
Tend = Tbgn + T - 1;

% X-dim index
Iend = (1:D)*N;
Ibgn = Iend - N + 1;

Xcov = zeros(N*D,N*D);

for i=1:D
  for j=1:i
  	XX = reshape(X(:,Tbgn(i):Tend(i),:), N,T*Nr) ...
  	   * reshape(X(:,Tbgn(j):Tend(j),:), N,T*Nr)';
  	   
  	Xcov(Ibgn(i):Iend(i), Ibgn(j):Iend(j)) = XX;
  	Xcov(Ibgn(j):Iend(j), Ibgn(i):Iend(i)) = XX';
  end
end

Xcov = (Xcov + Xcov')/2;
