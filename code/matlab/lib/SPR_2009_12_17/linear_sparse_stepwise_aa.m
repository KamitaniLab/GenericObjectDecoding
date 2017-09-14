function [Model, Info] = linear_sparse_stepwise(X,Y,Model,parm)
%  Estimate linear weight matrix for input-output mapping
%     Automatic Relevance Prior for each input dimension
%     is imposed to get sparse weight matrix
%
% Notice !!
%     Delay embedding is done in this module,
%     then input vector should be original input data without embedding
%
%   [Model, Info] = linear_sparse_stepwise(X,Y,Model,parm)
%     Scalar iteration version
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
%  Model.A   :  Prior weight variance  ( N x M*D ) 
%  Model.ix_act : Active index for W after pruning
%
%  Info  : Structure for learning process history
%  Info.FE  = LP + H : Free energy
%  Info.LP  = Log likelihood
%  Info.H   = - Model entropy
%
% 2009-10-18 Made by M. Sato

MINVAL = 1.0e-15;

fprintf('linear_sparse_stepwise start\n')

% Dimension
[N ,T  ,Ntrial]  = size(Y); % N = # of output
[M ,Tx ,Ntrialx] = size(X); % M = # of input without embedding

Tall = T * Ntrial;

if Ntrial~=Ntrialx, error('# of trial is different in X and Y'); end;

Nskip  = 100;   % skip steps for display info
a_min = 1e-14;	% Minimum value for weight pruning
Fdiff = 1e-12; % Threshold for convergence
Ncheck = 100;   % Minimum number of training iteration
Fstep  = 5;     % Free energy convergence check step
Prune  = 1;     % Prune mode
space_ARD=0;	% if space_ARD=1, ARD is done only for space dimension

if isfield(parm,'Nskip'), Nskip  = parm.Nskip; end;
if isfield(parm,'Fdiff'), Fdiff   = parm.Fdiff; end;
if isfield(parm,'a_min'), a_min   = parm.a_min ; end;
if isfield(parm,'Prune'), Prune = parm.Prune; end;
if isfield(parm,'Ncheck'), Ncheck = parm.Ncheck; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end
if isfield(parm,'Fstep'),Fstep  = parm.Fstep ; end
if isfield(parm,'space_ARD'),space_ARD  = parm.space_ARD ; end

% # of embedding dimension
if isfield(parm,'Dtau')
	D    = parm.Dtau; 
	tau  = parm.Tau;
else
	D    = 1;
	tau  = 1;
end
% Parameters
Ntrain = parm.Ntrain;

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

if isfield(parm,'Nupdate')
	Nupdate = parm.Nupdate;
else
	Nupdate = 1;
end

fprintf('--- Output Dimension  = %d\n',N)
fprintf('--- Input  Dimension  = %d\n',M)
fprintf('--- Embedding  Dimension  = %d\n',D)
fprintf('--- Number of trials  = %d\n',Ntrial)
fprintf('--- Number of training sample = %d\n',Tall)
fprintf('--- Total update iteration    = %d (%d)\n',Ntrain,Npre_train)

% original input dimension
Xdim  = M;
M = Xdim*D;
M_ALL = M;

%  
% --- Initialization
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

% Input covariance
XX  = sum(sum(X.^2,3),2)/(Tx*Ntrial);
XX  = repmat(XX', [1 D]);% 1 x (M*D)
XXX = repmat(XX,  [N 1]);% N x (M*D)

if isfield(Model,'ix_act')
	SY  = mean(Model.SY);  % 1 x 1

	% Active index
	ix_act = Model.ix_act;
	A = zeros(N,M_ALL);
	W = zeros(N,M_ALL);
	
	A(:,ix_act)  = Model.A;
	W(:,ix_act) = Model.W;

	if M_ALL ~= Xdim*D,
		fprintf('M_ALL=%d,Xdim=%d,D=%d\n',M_ALL ,Xdim ,D)
		error('M_ALL ~= Xdim*D')
	end
	
	% Active index in input space without embedding
	IX_dim = find( sum(reshape(A,[Xdim,D]),2) > 0);
	
	IX_act = repmat( (0:(D-1))* Xdim ,[length(IX_dim) 1]) ...
	       + repmat(IX_dim, [1 D]);
	IX_act = IX_act(:);
	
	W  = W(:,IX_act) ;  % N x (M*D)
	A  = A(:,IX_act) ;  % 1 x (M*D)
	X  = X(IX_dim,:,:);
	XX = XX(IX_act);  
	
	M     = length(IX_act);
	Xdim  = length(IX_dim);
else
	A = repmat(A0, [N,M_ALL]);
	W = zeros(N,M_ALL);
	SY  = SY0;
	% Save original input for pruning
	IX_act  = 1:M_ALL;
	IX_dim  = 1:Xdim;
end

ix_act  = 1:M;
ix_dim  = 1:Xdim;
M_all   = M;

% Delay embedding index for W
Wid = [(0:(D-1))'* Xdim + 1 , (1:D)'* Xdim];

% Working variable
G_A = zeros(N,M);       % N x M

SW = SY./( Tall*XXX + SY./A );

fprintf('a_min = %g\n', a_min)
fprintf('SY0 = %g\n', SY0)
fprintf('SY  = %g\n', SY)

% Free energy histry
FE  = zeros(Ntrain,1);
LP  = zeros(Ntrain,1);
H   = zeros(Ntrain,1);
Err = zeros(Ntrain,1);
Mhist = zeros(Ntrain,1);

% ARD hyper param. history
if isfield(parm,'Debug') && ~isempty(parm.Debug) && parm.Debug > 0
	Debug = 1;
	A_tmp = zeros(N*M_ALL, ceil(Ntrain/Nskip));
else
	Debug = 0;
end

% recover all component
A_all  = zeros(N,M_all);       % N x M

[dY] = error_delay_time(X,Y,W,tau);

k_save  = 0;

%%%%%% Learning Loop %%%%%%
for k=1:Ntrain
	% Ainv = alpha , A = 1/alpha
	Ainv    = SY./A;	
	
	% E = (Y-W*X)^2/SY +  W^2/A
	%   = ( (Y-W*X)^2 +  W^2 * (SY/A) )/SY
    [W]  = weight_update_embed_aa(X, dY, W, Tall*XX, Ainv, tau);
	[dY] = error_delay_time(X, Y, W, tau);

    dYY = sum(dY(:).^2)/(N*Tall); 
	WW  = W.^2;
    
    % Noise variance update
    SY  = dYY + sum(WW(:) .* Ainv(:))/(N*Tall);
    % Prevent zero variance
    SY  = max( SY, MINVAL);

    % Log variance
    SWA     = max( SW .* Ainv , MINVAL);
    log_sw  = N*(sum( log(SWA(:)) - SWA(:) + 1 ));
    log_sy  = N*( log(SY) );
    log_a   = Ta0*(sum( log(Ainv(:)) - a0.*Ainv(:) + 1 ));
	
    % Free energy
    LP(k)  = - (0.5*Tall) * log_sy ;
    H(k)   = 0.5*( log_sw + log_a - sum(WW(:) .*Ainv(:)) );
    FE(k)  = LP(k) + H(k);
    Err(k) = sum(dYY)/(SY0);

    % Weight variance

    SW = SY./( Tall*XXX + Ainv );
	
    % Hyper parameter for weight variance (ARD)
	if mod(k,Nupdate) == 0,
	    % A = Alpha^-1 * SY
		% Ainv = SY./A;	
	    % SW = SY./( Tall*XX + Ainv );
	    
	    % G_A = 1 - SW./A;
	    %     = (Tall*XX + Ainv - SY/A) * SW/SY
	    G_A = Tall.*XXX.*SW./SY;
	    G_A = max((G_A), MINVAL);
	    
		% N*A = N * (Alpha * SY) =  (W.^2) + N * SW  ; 
		if k <= Npre_train,
			% VB update rule (Stable)
			%  ARD for each weight
			A  = (WW + N*SW + 2*Ta0*a0)/( N + 2*Ta0 );
		else
			% Accelerated update rule (Unstable)
		    A  = sqrt( A.* WW./(G_A * N));
		    %A  = (WW + 2*Ta0*a0)./(G_A * N + 2*Ta0);
		end
		
		if space_ARD==1
			Am = mean(reshape(A,[N,M/D,D]),3);
			A  = repmat(Am, [1, D]);
		end
		
	    % Prune small variance
	    if Prune == 1
		    % Find active input dimension
		    ix_act_old = ix_act;
		    ix_dim_old = ix_dim;

		    % recover all component
			A_all  = zeros(N,M_all);       % 1 x M
		    %  A_all(:,ix_act)  = A;
		    A_all(:,ix_act) = WW/max(WW(:));
		    
		    % find active input dimension (absolute index)
		    ix_dim = find( sum(reshape(sum(A_all,1),[Xdim,D]),2) > a_min );
		    ix_act = repmat(ix_dim, [1 D]) ...
		           + repmat([0:D-1]*Xdim, [length(ix_dim) 1]);
		    ix_act = ix_act(:);
		    
		    Mnew   = length(ix_act);  		% # of effective input
		    
		    if Mnew < M,
			    % convert to relative index
			    jx_act = trans_index(ix_act,ix_act_old,M_all);
			    jx_dim = trans_index(ix_dim,ix_dim_old,Xdim);
			    
			    M   = Mnew;
			    A   = A(:,jx_act) ;  % N x M
			    W   = W(:,jx_act) ;  % N x M
			    SW  = SW(:,jx_act);  % N x M
	
			    X	 = X(jx_dim,:,:);  	 	% M x T
			    XX	 = XX(jx_act);  	 	% M x T
			    XXX	 = XXX(:,jx_act);  	 	% M x T
			end
	    end
	    % END of if Prune == 1

		A = max(A,MINVAL);
	    
	end
	% END of if mod(k,Nupdate) == 0
	
	Mhist(k) = M;
	SW  = SW /SY;

    if mod(k, Nskip)==0
        % Save history
		if Debug == 1
        	k_save = k_save + 1;
        	A_tmp(:,k_save) = A_all(:);
		end
		
        fprintf('Iter = %4d, M = %4d, err = %g, F = %g, SY= %g, H = %g\n', ...
               k, length(ix_act), Err(k), FE(k), SY, - 2*H(k)/log(T));
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

ix_act = IX_act(ix_act);

% Active index
Model.ix_act = ix_act;
Model.M_all  = M_ALL ;

% Save output variable
Model.A    = A ;
Model.W    = W ;
Model.SY   = SY;
Model.SW   = SW; % = 1./(Tall*XX + diag(Ainv))

Model.method = 'linear_sparse_stepwise';
Model.mode   = 'scalar';
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
