function [z ,t] = delay_embed(x,tau,D,T)
% Make time delayed embedding vector
%  [z ,t] = delay_embed(x,tau,D,T)
%  z(:,t) = [ x(:,t); x(:,t-tau); ...; x(:,t-tau*(D-1) ]
%
% --- Input
%  x    : x(:,t) , t=1:Tsample  [ N x Tsample x Ntrial]
%  tau  : Lag time
%  D    : Number of time delay
% --- Optional Input
%  T    : Output Time length
%
% --- Output
% z(:,t) = [ x(:,t); x(:,t-tau); ...; x(:,t-tau*(D-1) ]
%    t   = (1:T) + Tdelay;
% Tdelay = tau*(D-1);
% Condition:  Tsample >= T + Tdelay
%
% 2007/1/15 Made by M. Sato

[N ,Tall, Ntrial] = size(x);

if D < 2, 
	z = x;
	t = (1:Tall);
	return
end;

Tdelay = tau*(D-1);

if ~exist('T','var')
	T = Tall - Tdelay;
elseif Tall < (T + Tdelay),
	T = Tall - Tdelay;
	fprintf('Number of data is smaller than %d\n', (T + Tdelay))
end

% Time delayed embedding index

indxT = (1:T) + Tdelay;
indxD = 0:tau:Tdelay;
indx  = repmat( -indxD(:),[1 T]) + repmat(indxT,[D 1]);
indx  = indx(:);

z = x(:, indx, :);
z = reshape( z, [N*D T Ntrial]);
t = indxT;

return

% --- equivalent calculation
%     Above index calculation is much faster than the following

ix = 1:N;
t  = 1:T;

z  = zeros(N*D,T);

for n=1:D,
  z(ix, :) = x(:,t);
  ix = ix + N;
  t  = t  + tau;
end;

