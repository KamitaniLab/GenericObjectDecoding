function [Model, Info] = linear_sparse_space_sw(X,Y,Model,parm)
%  Estimate linear weight matrix for input-output mapping
%     Automatic Relevance Prior for each input dimension 
%     in original input space (not for delay embedding space)
%     is imposed to get sparse weight matrix
%  Trial data with variable sample number is supported 
%     by setting sample number of each trial
%  Notice !!
%     Delay embedding is done in this module,
%     then input vector should be original input data without embedding
%
%     Covariance is calculated in original input space,
%     then increse of embedding dimension make no problem in computation
% 
%   [Model, Info] = linear_sparse_space_sw(X,Y,Model,parm)
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
%  parm.Trial : Valid_sample number for each trial [Ntrial x 1]
%    if parm.Trial = [15 10] and T = 15, 
%       last 5 samples in 2nd trial is not used for estimation
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

fprintf('Linear sparse space (variable trial length) start\n')

% Dimension
[N ,T  ,Ntrial]  = size(Y); % N = # of output
[M ,Tx ,Ntrialx] = size(X); % M = # of input without embedding

% Number of samples for each trial
if isfield(parm,'Trial'), 
	Trial = parm.Trial;
	if length(Trial) ~= Ntrial, error('Trial sample seting is wrong');end
	if max(Trial) > T, error('Trial sample seting is wrong');end
else
	Trial = repmat(T, [Ntrial 1]);
end
% Set valid sample flag for each trial
trial_sw = zeros(N,T,Ntrial);
for n = 1:Ntrial
	trial_sw(:,1:Trial(n),n) = 1;
end
	
Tall = sum(Trial);

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

if Ntrain < 1, Info = []; return; end;

if isfield(parm,'Npre_train')
	Npre_train = parm.Npre_train;
else
%	Npre_train = Ntrain;
	if Tall >= 2*M*D
		Npre_train = 0;
	elseif Tall >= M*D
		Npre_train = fix(Ntrain/2);
	else
		Npre_train = Ntrain;
	end
end

if Npre_train > Ntrain, Npre_train = Ntrain; end;

fprintf('--- Output Dimension  = %d\n',N)
fprintf('--- Input  Dimension  = %d\n',M)
fprintf('--- Embedding  Dimension  = %d\n',D)
fprintf('--- Number of trials  = %d\n',Ntrial)
fprintf('--- Number of training sample = %d\n',Tall)
fprintf('--- Total update iteration    = %d (%d)\n',Ntrain,Npre_train)

% Original input dimension
Xdim  = M;
M_ALL = Xdim*D;

%  
% --- Initialization
%  W , A : Initial variable to use 1st update
%  

% Input/Output variance
sx = mean((X(:) - mean(X(:))).^2);
sy = mean((Y(:) - mean(Y(:))).^2);

SY0 = mean(sy);
A0  = 1./mean(sx);

if isfield(parm,'Ta0') && parm.Ta0 > 0,
	Ta0 = parm.Ta0;
	a0  = parm.a0 * A0;
else
	Ta0 = 0;
	a0  = 1;
end

% 'linear_sparse_space'
%    ARD term = alpha * W^2  : A = W^2
if isfield(Model,'ix_act')
	SY  = mean(Model.SY);  % 1 x 1

	% Recover old estimate of A in full embedding space
	ix_act = Model.ix_act;
	Aold = zeros(1,M_ALL); % 1 x M_ALL
	Wold = zeros(N,M_ALL);
	
	Aold(ix_act) = sum(Model.A,1);
	Wold(:,ix_act) = Model.W;

	if M_ALL ~= Xdim*D,
		fprintf('M_ALL=%d,Xdim=%d,D=%d\n',M_ALL ,Xdim ,D)
		error('M_ALL ~= Xdim*D')
	end
	% Summation over time delay component
	A  = mean(reshape(Aold,[Xdim,D]),2);	% Xdim x 1
	
	% Active index in input space without embedding
	IX_act = find(A > 0);
	A  = A(IX_act);
	M  = length(IX_act);
	
	id  = repmat( (0:(D-1))* Xdim ,[M 1]) + repmat(IX_act, [1 D]);
	W   = Wold(:,id(:)) ;  % N x (M*D)

	X  = X(IX_act,:,:); % M x T
else
	A = repmat(A0, [Xdim, 1]);
	W = zeros(N,M_ALL);
	SY  = SY0;
	% Save original input index for pruning
	IX_act = (1:Xdim)';
end

rx  = 1/(D + 1);
SX  = SY*rx;

fprintf('a_min = %g\n', a_min)
fprintf('SY0 = %g\n', SY0)
fprintf('SY  = %g\n', SY)

% Active index in input space without embedding
ix_act  = (1:M)';

% Delay embedding index for W
Wid = [(0:(D-1))'* M + 1 , (1:D)'* M];

% Input covariance (Spatial dimension)
Trial = Trial+(D-1)*tau;

XX = zeros(M,M);
for n=1:Ntrial
	XX = XX + X(:,1:Trial(n), n) * X(:,1:Trial(n), n)';
end
XX  = XX / sum(Trial);

SW = inv( Tall.* XX + diag(1./A) );

% Working variable
dY  = zeros(N,T,Ntrial);% N x T x Ntrial
dYX = zeros(N,M*D);     % N x M*D
G_A = zeros(M,1);       % M x 1
WW  = sum(reshape(sum(W.^2,1),[M,D]),2);	% M x 1

% Free energy histry
FE  = zeros(Ntrain,1);
LP  = zeros(Ntrain,1);
H   = zeros(Ntrain,1);
Err = zeros(Ntrain,1);
Mhist = zeros(Ntrain,1);

% ARD hyper param. history
if isfield(parm,'Debug') && ~isempty(parm.Debug) && parm.Debug > 0
	Debug = 1;
	A_tmp = zeros(M_ALL, ceil(Ntrain/Nskip));
else
	Debug = 0;
end
% recover all component
A_all  = zeros(M,1);

k_save  = 0;

%%%%%% Learning Loop %%%%%%
for k=1:Ntrain
	% Ainv = alpha , A = 1/alpha
	Ainv    = 1./A;	

	%    dY  = Y - W * X
	%    dYX = (dY * X')/T;
	dY  = error_delay_time_sw(X, Y, W, tau, Trial); % N x T x Ntrial
	dY  = dY .* trial_sw;                           % mask invalid samples
    dYX = error_corr_delay_sw(X, dY, D, tau, Trial);% N x M*D
    
    dYY = sum(dY(:).^2)/(N*Tall); 
    dYX = dYX/(Tall);   

    % Noise variance update
    WWA = sum(WW .*Ainv);
    SX  = (rx) .* dYY + WWA/(N*Tall) ;
	%  SX  = rx .* dYY + sum(SW(:) .* XX(:)) ./rx ;

    % Prevent zero variance
    SX  = max( SX, MINVAL);
	SY  = SX/rx;
	
    % Log variance
    SWA     = diag(SW) .* Ainv;
    log_sw  = (N*D)*(log_det(SW) + sum(log(Ainv)) - sum(SWA) + M);
    log_sy  = N * sum( log(SY) );
    log_a   = Ta0*sum(log(Ainv) - a0.*Ainv + 1);
    
    % Free energy
    LP(k)  = - (0.5*Tall) * log_sy ;
    H(k)   = 0.5*( log_sw + log_a );
%    H(k)   = 0.5*( log_sw + log_a - WWA );
    FE(k)  = LP(k) + H(k);
    Err(k) = sum(dYY)/(SY0);

    % Weight variance
    % SW = inv( (T./SX) .* XX + diag(Ainv) );   ( M x M )
    SW  = XX + diag( Ainv./Tall );
    
   	if rcond(SW) > MinCond,
		SW  = inv( SW );
	else
		SW  = pinv( SW );
	end

    % Weight update
    % W   = (T/SX)*( W * XX + rx .* dYX ) * SW; ( N x MD )
	for j=1:D
    	W(:,Wid(j,1):Wid(j,2)) = ( W(:,Wid(j,1):Wid(j,2)) * XX ...
    	                       + rx * dYX(:,Wid(j,1):Wid(j,2)) ) * SW;
    end
    
	%  ARD for each input (average over output & delay)
	WW  = sum(reshape(sum(W.^2,1),[M,D]),2);		% M x 1

    % Hyper parameter for weight variance (ARD)
    SW  = SW /Tall; % = inv(Tall*XX + diag(Ainv))
%  	G_A = 1 - diag(SW).*Ainv ;
%  	    = diag( ((Tall*XX + Ainv) - Ainv) * SW )
%  	    = diag( (Tall*XX) * SW ) 
  	G_A = Tall*sum(XX.*SW,2);
	G_A = max((G_A), MINVAL);
	
	if k <= Npre_train,
		% VB update rule (Stable)
		% (N*D)*A  = WW./SX + (N*D) * diag(SW) 
		% (N*D)*A*(1 - diag(SW)./A)  = WW./SX 
		% Modefied VB update rule (Stable)
		A  = (WW./SX + (N*D) * diag(SW) + 2*Ta0*a0)/( N*D + 2*Ta0 );
%	    A  = sqrt(A.*(WW./SX)./(G_A*N*D));
	else
	    % Accelerated update rule
	    A  = sqrt(A.*(WW./SX)./(G_A*N*D));
%	    A  = ((WW./SX) + 2*Ta0*a0)./(G_A*N*D + 2*Ta0);
	end
	
    % Prune small variance
    if Prune > 0
	    % Find active input dimension
	    ix_act_old = ix_act;
	    
	    % Recover all component
	    switch	Prune
	    case	1
		    A_all(ix_act) = WW/max(WW);    % Prune by Weight
	    case	2
		    A_all(ix_act) = A /max(A);	   % Prune by Alpha
	    case	3
		    A_all(ix_act) = A * (1/SX);    % Prune by Alpha
	    end
	    
	    % Find active input dimension (absolute index)
	    ix_act = find( A_all > a_min ); % effective indices
	    Mnew   = length(ix_act);  		% # of effective input
	    
	    if Mnew < M,
		    % convert to relative index
		    jx_act = trans_index(ix_act,ix_act_old,Xdim);
		    
		    A   = A(jx_act) ;  % 1 x M
		    SW  = SW(jx_act,jx_act);  % M x M

			% Effective delay embedding index for W
			id  = repmat( (0:(D-1))* M ,[Mnew 1]) + repmat(jx_act, [1 D]);
		    W   = W(:,id(:)) ;  % N x MD

			% New delay embedding index for W
		    M   = Mnew;
			Wid = [(0:(D-1))'* M + 1 , (1:D)'* M];
			
			WW  = WW(jx_act);
		    X   = X(jx_act,:,:);  	 % M x T x Ntrial
		    XX  = XX(jx_act,jx_act); % M x M
		end
    else
		A = max(A,MINVAL);
    end
    % END of if Prune == 1
	Mhist(k) = M;
	
    if mod(k, Nskip)==0
        % Save history
		if Debug == 1
        	k_save = k_save + 1;
        	A_tmp(:,k_save) = A_all(:);
		end
		
        fprintf('Iter=%4d, M =%4d, err=%g, F=%g, H=%g, SY=%g, Wmin=%g\n',...
        		        k, M, Err(k), FE(k), -2*H(k)/log(Tall), ...
        		        SY, min(WW)/max(WW));
    end
    
    % Convergence check
	if k > Ncheck,
		Fdif = (FE(k) - FE(k-Fstep))/(abs(FE(k))+eps);
	else
		Fdif = Fdiff + 1;
	end
	
	if (Fdiff > abs(Fdif)), 
		fprintf('Converged : Free energy change = %g\n',Fdif)
		break; 
	end;
		
end

% convert to relative index
ix_act = IX_act(ix_act);

% Delay embedding index for W in full input space
id  = repmat( (0:(D-1))* Xdim ,[M 1]) + repmat(ix_act, [1 D]);
% Recover A in full embedding input space
A   = repmat(A(:), [1 D]);

% Active index
Model.ix_act = id(:);
Model.M_all  = M_ALL ;

% Save output variable
Model.A    = A(:)' ; % 1 x M*D
Model.W    = W ;
Model.SY   = SY;
Model.SW   = SW; % = inv(Tall*XX + diag(Ainv))

Model.method = 'linear_sparse_space';
Model.mode   = 'cov';
Model.sparse = 'sparse';

% Save history
Info.FE  = FE(1:k);
Info.LP  = LP(1:k);
Info.H   = H(1:k) ;
Info.Err = Err(1:k);
Info.M   = Mhist(1:k);

if exist('A_tmp','var')
	Info.A   = A_tmp(:,1:k_save) ;
end

%%% ---- Index transformation from old active_index to current active_index
function	jx = trans_index(ix,ix_old,M)

N = length(ix_old);
Itrans = zeros(M,1);
Itrans(ix_old) = 1:N;

jx = Itrans(ix);
