function [Model, Info] = linear_sparse_cov(X,Y,Model,parm)
%  Estimate linear weight matrix for input-output mapping
%     Automatic Relevance Prior for each input dimension
%     is imposed to get sparse weight matrix
%
% Notice !!
%     Delay embedding is done in this module,
%     then input vector should be original input data without embedding
%
%   [Model, Info] = linear_sparse_cov(X,Y,Model,parm)
%
% --- Input
%  Y  : Output data ( N x T x Ntrial )
%  X  : Input data  ( M x (T + (Dtau-1)*Tau)  x Ntrial)
%  N  =  # of output
%  M  =  # of input (original input space dimension)
%  T  =  # of time sample
%
%  Estimate the following model
%    Y(t) = W * [X(:, t + (Dtau-1)*Tau ); ...; X(:, t)]
%
%  Model : Structure for estimated model
%  if Model is empty, initialization is done before training
%  if Model is previous training result, re-training us done
%
%  parm  : Structure for learning parameter
%  parm.Npre_train :  # of VB-update in initial training
%  parm.Ntrain :  # of training
%  parm.Nskip  :  skip # for print
%  parm.a_min  :  Min value for pruning small variance component
%  parm.Prune  :  = 1 : Prune small variance & irrelevant input dimension
%
%  parm.Tau       = Lag time
%  parm.Dtau      = Number of embedding dimension
%    Total input dimension in embedding space is (M * parm.Dtau)
% --- Output
%  Model : Structure for estimated model
%  Model.SY  :  Noise variance         ( 1 x 1 )
%  Model.W   :  Weight matrix          ( N x M*D ) , D = parm.Dtau
%  Model.A   :  Prior weight variance  ( 1 x M*D ) 
%  Model.ix_act : Active index for W after pruning
%
%  Info  : Structure for learning process history
%  Info.FE  = LP + H : Free energy
%  Info.LP  = Log likelihood
%  Info.H   = - Model entropy
%
% 2009-10-18 Made by M. Sato

MINVAL = 1.0e-15;
MinCond = 1e-8;

% Dimension
[N ,T  ,Ntrial]  = size(Y); % N = # of output
[M ,Tx ,Ntrialx] = size(X); % M = # of input without embedding

Tall = T * Ntrial;

if Ntrial~=Ntrialx, error('# of trial is different in X and Y'); end;

% # of total training iteration
Ntrain = parm.Ntrain;

Nskip  = 100;   % skip steps for display info
a_min  = 1e-10; % Minimum value for weight pruning
Fdiff  = 1e-10; % Threshold for convergence
Ncheck = 100;   % Minimum number of training iteration
Fstep  = 5;     % Free energy convergence check step
Prune  = 1;     % Prune mode

if isfield(parm,'Nskip'), Nskip  = parm.Nskip; end;
if isfield(parm,'Fdiff'), Fdiff   = parm.Fdiff; end;
if isfield(parm,'a_min'), a_min   = parm.a_min ; end;
if isfield(parm,'Prune'), Prune = parm.Prune; end;
if isfield(parm,'Ncheck'), Ncheck = parm.Ncheck; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end

% # of embedding dimension
if isfield(parm,'Dtau')
	D    = parm.Dtau; 
	tau  = parm.Tau;
else
	D    = 1;
	tau  = 1;
end

if isfield(parm,'Npre_train')
	Npre_train = parm.Npre_train;
else
	Npre_train = Ntrain;
end

if Npre_train > Ntrain, Npre_train = Ntrain; end;

fprintf('linear sparse covariance start\n')
fprintf('--- Output Dimension  = %d\n',N)
fprintf('--- Input  Dimension  = %d\n',M)
fprintf('--- Embedding  Dimension  = %d\n',D)
fprintf('--- Number of trials  = %d\n',Ntrial)
fprintf('--- Number of training sample = %d\n',Tall)
fprintf('--- Total update iteration    = %d (%d)\n',Ntrain,Npre_train)

%  
% --- Initialization
%

% Input/Output variance
sx = mean((X(:) - mean(X(:))).^2);
sy = mean((Y(:) - mean(Y(:))).^2);

SY0 = mean(sy);
A0  = SY0./mean(sx);

% Original input dimension
Xdim  = M;
M_ALL = Xdim*D;

% 'linear_sparse_cov'
%    ARD term = alpha * W^2 * SY(^-1) : A = W^2/SY
if isfield(Model,'ix_act')
	SY  = mean(Model.SY);  % 1 x 1
	
	% Recover old estimate of A in full embedding space
	ix_act = Model.ix_act;
	A = zeros(1,M_ALL);
	W = zeros(N,M_ALL);
	
	A(ix_act)   = sum(Model.A,1)/SY;
	W(:,ix_act) = Model.W;

	if M_ALL ~= Xdim*D,
		fprintf('M_ALL=%d,Xdim=%d,D=%d\n',M_ALL ,Xdim ,D)
		error('M_ALL ~= Xdim*D')
	end
	
	% Active index in input space without embedding
	ix = find( sum(reshape(A,[Xdim,D]),2) > 0);
	
	IX_act = repmat( (0:(D-1))* Xdim ,[length(ix) 1]) + repmat(ix, [1 D]);
	IX_act = IX_act(:);
	W  = W(:,IX_act) ;  % N x (M*D)
	A  = A(IX_act) ;  % 1 x (M*D)
	X  = X(ix,:,:); % M x T
	
	Xdim = length(ix);
	M = length(IX_act);
else
	A = repmat(A0, [1,M_ALL]);
	W = zeros(N,M_ALL);
	SY  = SY0;
	M = M_ALL;
	IX_act = 1:M;
end

% Active index in input space without embedding
ix_act = (1:M)';
ix_dim = 1:Xdim;
M_all  = M;

A   = max(A,MINVAL);

% Input variance
XX  = embed_covariance(X,D,tau);
YX  = error_corr_delay(X,Y,D,tau);
YY  = sum(Y(:).^2);

%fprintf('XXmin = %g\n', min(diag(XX)))
%fprintf('XXmax = %g\n', max(diag(XX)))
%fprintf('Wmin = %g\n', min(sum(W.^2,1)))
%fprintf('Wmax = %g\n', max(sum(W.^2,1)))
%fprintf('Amin = %g\n', min(A))
%fprintf('Amax = %g\n', max(A))
fprintf('a_min = %g\n', a_min)
fprintf('SY0   = %g\n', SY0)
fprintf('SY    = %g\n', SY)

fprintf('Start Input Cov dim = %d x %d \n',size(XX))

% Working variable
SW  = zeros(M,M);       % M x M
AA  = zeros(M,M);       % M x M
G_A = zeros(1,M);       % 1 x M
log_a = 0;

if isfield(parm,'Ta0') && parm.Ta0 > 0,
	Ta0 = parm.Ta0;
	a0  = parm.a0 * A0;
else
	Ta0 = 0;
	a0  = 1;
end

% Free energy histry
FE  = zeros(Ntrain,1);
LP  = zeros(Ntrain,1);
H   = zeros(Ntrain,1);
Err = zeros(Ntrain,1);
MM  = zeros(Ntrain,1);

% ARD hyper param. history
if isfield(parm,'Debug') && ~isempty(parm.Debug) && parm.Debug > 0
	Debug = 1;
	A_tmp = zeros(M_ALL, ceil(Ntrain/Nskip));
else
	Debug = 0;
end

A_all = zeros(1,M);

k_save  = 0;

%%%%%% Learning Loop %%%%%%
for k=1:Ntrain
	% ARD hyper variance parameter
	% A = 1/alpha, Ainv = alpha 

    % Weight variance
	SW  = XX + diag(1./A);
	if rcond(SW) > MinCond,
		SW  = inv( SW );  % M x M
	else
		SW  = pinv( SW );
	end

	G_A = sum( SW .* XX );			% 1 x M

	% Weight update
	W  = YX * SW;
   	WW = sum(W.^2,1);
	
    % Noise variance update: Error
    SY = (YY - sum(sum(W.*YX)))/(N*Tall);
    
    if (SY/SY0) <= MINVAL,
	    % dY  = Y - W * X;        	% N x T
	    dY  = error_delay_time(X, Y, W, tau);
	    dYY = sum(dY(:).^2);  	% 1 x 1
	
	    SY  = (dYY + sum( WW ./ A ))/(Tall*N);	% 1 x 1
	    % Prevent zero variance
	    SY  = max( SY, MINVAL);
	    fprintf('*')
    end

	% Log variance
	log_sw  = log_det(SW) - sum(log(A));
    log_sy  = log(SY) ;
    log_a   = Ta0 * sum( - log(A) - a0./A + 1);
	
    % Free energy
    H(k)   =   (0.5*N) * (log_sw - M) + 0.5*log_a;
    LP(k)  = - (0.5*N*Tall) * log_sy ;
    FE(k)  = LP(k) + H(k);
    Err(k) = (SY)/(SY0);
    MM(k)  = M;

    % Hyper parameter for weight variance (ARD)
    G_A = max((G_A), MINVAL);

	if k <= Npre_train,
		% VB update rule
		%  	N * A  = (1./SY)' * (W.^2) + N * (A - A.*G_A)  ; 
   		%A  = WW./SY + N * (A - A.*G_A)  ; 
		%A  = (A + 2*Ta0*a0)./( N + 2*Ta0 );
		%	A^2  = A .* (1./SY)' * (W.^2) ./ (G_A * N);	
	    A  = sqrt(A.*(WW./SY)./(G_A * N));
	else
	    % Accelerated update rule
		%	A  = (1./SY)' * (W.^2) ./ (G_A * N);	
	    A  = ((WW./SY) + 2*Ta0*a0)./(G_A * N + 2*Ta0);
	end
	
    % Prune small variance
    if Prune == 1
		% Find active input dimension
		ix_act_old = ix_act;
		ix_dim_old = ix_dim;

		% recover all component
		A_all  = zeros(1,M_all);       % N x M
%		A_all(:,ix_act)  = A;
		A_all(ix_act) = WW/max(WW);
		
		% find active input dimension (absolute index)
		ix_dim = find( sum(reshape(A_all,[Xdim,D]),2) > a_min );
		ix_act = repmat(ix_dim, [1 D]) ...
		       + repmat([0:D-1]*Xdim, [length(ix_dim) 1]);
		ix_act = ix_act(:);
		
		Mnew   = length(ix_act);  		% # of effective input
		
		if Mnew < M,
		    % convert to relative index
		    jx_act = trans_index(ix_act,ix_act_old,M_all);
		    jx_dim = trans_index(ix_dim,ix_dim_old,Xdim);
		    
		    M   = Mnew;
		    A   = A(jx_act) ;  % N x M
		    W   = W(:,jx_act) ;  % N x M
		    SW  = SW(jx_act);  % N x M
	
		    X	 = X(jx_dim,:,:);  	 	% M x T
			YX  = YX(:,jx_act);  	 	% N x M
			XX	= XX(jx_act,jx_act);	% M x M
		end
    end

	A = max(A,MINVAL);

    if mod(k, Nskip)==0
        % Save history
		if Debug == 1
        	k_save = k_save + 1;
        	A_tmp(:,k_save) = A_all(:);
		end
		
        fprintf('Iter = %4d, M = %4d, err = %g, F = %g, H = %g\n', ...
               k, M, Err(k), FE(k), - H(k));
    end


	if k > Ncheck && M == MM(k-1)
		Adif = max(abs(A - A_old));
	else
		Adif = 1;
	end
	if Adif < Fdiff, 
		fprintf('Converged : Alpha change = %g\n',Adif)
		break; 
	end;
	
	A_old = A;
	
%	if k > Ncheck,
%		Fdif = (FE(k) - FE(k-Fstep))/abs(FE(k));
%	else
%		Fdif = Fdiff + 1;
%	end
%	
%	if (Fdiff > abs(Fdif)), 
%		fprintf('Converged : Free energy change = %g\n',Fdif)
%		break; 
%	end;
end

% convert to relative index
ix_act = IX_act(ix_act);

% Active index
Model.ix_act = ix_act;
Model.M_all  = M_ALL ;

% Save trained variable
%  W & A is sufficient for cov-method initialization
Model.A  = A * SY ;
Model.W  = W ;
Model.SY = SY;
Model.SW = SW; % = inv(Tall*XX + diag(Ainv))
% E = ( (Y-W*X)^2 +  W^2 * (1/A) )/SY

Model.method = 'linear_sparse_cov';
Model.mode   = 'cov';
Model.sparse = 'sparse';

% Save history
Info.FE  = FE(1:k);
Info.LP  = LP(1:k);
Info.H   = H(1:k) ;
Info.Err = Err(1:k);
Info.M   = MM(1:k);

if exist('A_tmp','var')
	Info.A   = A_tmp(:,1:k_save) ;
end


% recover all component
%W_all   = zeros(N,M_all);
%SW_all  = zeros(N,M_all);
%
%A_all(:,ix_act)  = A;
%W_all(:,ix_act)  = W ;  % N x M
%SW_all(:,ix_act) = SW;  % N x M

%%% ---- Index transformation from old active_index to current active_index
function	jx = trans_index(ix,ix_old,M)

N = length(ix_old);
Itrans = zeros(M,1);
Itrans(ix_old) = 1:N;

jx = Itrans(ix);
